"""Observation-conditioned Gaussian frontends and a small, ordinary DPLR S4.

No hidden historical truth is accepted by this module. A latent describes an
entire window; prefix posteriors do not imply a dynamic Kalman transition.
"""
from __future__ import annotations

import importlib
import math
import pathlib
import sys
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


EXP_ROOT = pathlib.Path(__file__).resolve().parents[1]
S4_ROOT = EXP_ROOT / "baselines/S4M/s4m/model/S4/s4"
VARIANTS = ("LI-S4", "STAT-S4", "PU-S4")


def s4_module():
    # ``extensions`` is a namespace package under this root. Merely importing
    # S4 through the outer S4M package misses the optional compiled kernels.
    if str(S4_ROOT) not in sys.path:
        sys.path.insert(0, str(S4_ROOT))
    return importlib.import_module("models.s4.s4")


@dataclass(frozen=True)
class ModelConfig:
    channels: int
    seq_len: int = 96
    pred_len: int = 96
    rank: int = 16
    width: int = 64
    state_size: int = 32
    layers: int = 2
    ff_width: int = 128
    dropout: float = 0.1
    noise_floor: float = 0.01

    def to_dict(self):
        return asdict(self)


def interpolate_history(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """GPU interpolation; endpoints extend the closest observed value.

    All-missing channels use zero (the training mean). The main protocol
    always observes exactly 48 positions, so this fallback is only defensive.
    """
    observed = mask.bool()
    x = torch.where(observed, x, torch.zeros_like(x))
    length = x.shape[1]
    pos = torch.arange(length, device=x.device).view(1, length, 1).expand_as(mask)
    left = torch.where(observed, pos, -1).cummax(1).values
    right = torch.where(observed, pos, length).flip([1]).cummin(1).values.flip([1])
    lv = x.gather(1, left.clamp(0, length - 1))
    rv = x.gather(1, right.clamp(0, length - 1))
    alpha = (pos - left).to(x.dtype) / (right - left).clamp_min(1).to(x.dtype)
    filled = torch.where((left >= 0) & (right < length), lv + alpha * (rv - lv),
                         torch.where(left >= 0, lv, rv))
    return torch.where(observed, x, filled)


class GaussianObservation(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.loadings = nn.Parameter(torch.randn(cfg.seq_len, cfg.channels, cfg.rank) * (0.1 / math.sqrt(cfg.rank)))
        self.mean = nn.Parameter(torch.zeros(cfg.channels))
        initial = math.log(math.expm1(1.0 - cfg.noise_floor))
        self.raw_variance = nn.Parameter(torch.full((cfg.channels,), initial))
        self.register_buffer("eye", torch.eye(cfg.rank))

    @property
    def variance(self):
        return F.softplus(self.raw_variance) + self.cfg.noise_floor

    def forward(self, x, mask, posterior: bool = True, return_details: bool = False):
        # Mask before subtracting the mean; masked arbitrary values/NaNs
        # cannot contribute to the likelihood or information vector.
        residual = torch.where(mask.bool(), x - self.mean, torch.zeros_like(x))
        variance = self.variance
        a = self.loadings
        weighted_a = a / variance.view(1, -1, 1)
        # Intermediate L*C*r*r, never B*L*C*r*r.
        outer = (a.unsqueeze(-1) * weighted_a.unsqueeze(-2)).flatten(2)
        increments = torch.bmm(mask.transpose(0, 1), outer).transpose(0, 1)
        J = increments.reshape(*mask.shape[:2], self.cfg.rank, self.cfg.rank).cumsum(1) + self.eye
        b = torch.bmm(residual.transpose(0, 1), weighted_a).transpose(0, 1).cumsum(1)
        # No silent jitter: J includes I and must be positive definite.
        if posterior or return_details:
            chol = torch.linalg.cholesky(J)
            m = torch.cholesky_solve(b.unsqueeze(-1), chol).squeeze(-1)
            covariance = torch.cholesky_inverse(chol)
            features = torch.cat((m, covariance.diagonal(dim1=-2, dim2=-1).clamp_min(1e-12).log()), -1)
            last_chol, last_mean = chol[:, -1], m[:, -1]
        else:
            last_chol = torch.linalg.cholesky(J[:, -1])
            last_mean = torch.cholesky_solve(b[:, -1, :, None], last_chol).squeeze(-1)
            features = torch.cat((b, J.diagonal(dim1=-2, dim2=-1).clamp_min(1e-12).log()), -1)
            m = covariance = None
        count = mask.sum((1, 2))
        logdet = 2 * last_chol.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        quadratic = (residual.square() / variance).sum((1, 2))
        log_noise = (mask * variance.log()).sum((1, 2))
        nll = 0.5 * (count * math.log(2 * math.pi) + log_noise + logdet + quadratic
                     - (b[:, -1] * last_mean).sum(-1)) / count.clamp_min(1)
        if return_details:
            return features, nll, {"precision": J, "information": b, "mean": m, "covariance": covariance}
        return features, nll


class S4ResidualBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.s4 = s4_module().S4Block(
            cfg.width, transposed=False, d_state=cfg.state_size, mode="dplr",
            backend="cuda", lr=1e-3, dropout=cfg.dropout, verbose=False,
        )
        self.norm1 = nn.LayerNorm(cfg.width)
        self.norm2 = nn.LayerNorm(cfg.width)
        self.ff = nn.Sequential(nn.Linear(cfg.width, cfg.ff_width), nn.GELU(),
                                nn.Dropout(cfg.dropout), nn.Linear(cfg.ff_width, cfg.width), nn.Dropout(cfg.dropout))
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = self.norm1(x + self.drop(self.s4(x)[0]))
        return self.norm2(x + self.ff(x))


class PosteriorS4(nn.Module):
    def __init__(self, variant: str, cfg: ModelConfig, seed: int):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant, self.cfg = variant, cfg
        # All common weights start identically. Frontend creation must not
        # shift the S4 initialization or subsequent minibatch RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.blocks = nn.ModuleList([S4ResidualBlock(cfg) for _ in range(cfg.layers)])
            self.time_head = nn.Linear(cfg.seq_len, cfg.pred_len)
            self.channel_head = nn.Linear(cfg.width, cfg.channels)
            torch.manual_seed(seed + 1000)
            self.observation = None if variant == "LI-S4" else GaussianObservation(cfg)
            input_size = 2 * cfg.channels if variant == "LI-S4" else 2 * cfg.rank
            torch.manual_seed(seed + 2000)
            self.input_projection = nn.Linear(input_size, cfg.width)

    def forward(self, x, mask):
        if self.observation is None:
            features = torch.cat((interpolate_history(x, mask), mask), -1)
            nll = x.new_zeros(x.shape[0])
        else:
            features, nll = self.observation(x, mask, posterior=self.variant == "PU-S4")
        hidden = self.input_projection(features)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.time_head(hidden.transpose(1, 2)).transpose(1, 2)
        return self.channel_head(hidden), nll

    def parameter_counts(self):
        return {"total": sum(p.numel() for p in self.parameters()),
                "frontend": sum(p.numel() for p in self.input_projection.parameters()) +
                (0 if self.observation is None else sum(p.numel() for p in self.observation.parameters())),
                "s4_blocks": sum(p.numel() for p in self.blocks.parameters()),
                "head": sum(p.numel() for module in (self.time_head, self.channel_head) for p in module.parameters())}


def make_optimizer(model, lr=1e-3):
    groups = {}
    names = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        options = {"lr": lr, "weight_decay": 1e-4}
        options.update(getattr(parameter, "_optim", {}))
        options["lr"] = min(options["lr"], lr)
        key = tuple(sorted(options.items()))
        groups.setdefault(key, []).append(parameter)
        names.setdefault(key, []).append(name)
    optimizer = torch.optim.AdamW([dict(params=params, **dict(key)) for key, params in groups.items()])
    metadata = [{**dict(key), "names": names[key], "numel": sum(p.numel() for p in params)} for key, params in groups.items()]
    return optimizer, metadata
