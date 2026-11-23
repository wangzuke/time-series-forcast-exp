"""History-preserving OCPE fusion. Frozen legacy modules are imported read-only."""
from __future__ import annotations

import math
import torch
from torch import nn
from posterior_s4 import (GaussianObservation, ModelConfig, S4ResidualBlock,
                          interpolate_history, make_optimizer, s4_module)

VARIANTS = ("F0", "F1", "F2", "F3")
LABELS = {"F0": "LI-S4", "F1": "LI+DIAG-S4", "F2": "LI+MEAN-S4", "F3": "LI+MEANVAR-S4"}


class ConditionalEncoder(GaussianObservation):
    def forward(self, x, mask, mode="F3", return_details=False):
        residual = torch.where(mask.bool(), x - self.mean, torch.zeros_like(x))
        variance, a = self.variance, self.loadings
        weighted_a = a / variance.view(1, -1, 1)
        outer = (a.unsqueeze(-1) * weighted_a.unsqueeze(-2)).flatten(2)
        increments = torch.bmm(mask.transpose(0, 1), outer).transpose(0, 1)
        J = increments.reshape(*mask.shape[:2], self.cfg.rank, self.cfg.rank).cumsum(1) + self.eye
        b = torch.bmm(residual.transpose(0, 1), weighted_a).transpose(0, 1).cumsum(1)
        covariance, mean = None, None
        if mode == "F1" and not return_details:
            chol_last = torch.linalg.cholesky(J[:, -1])
            mean_last = torch.cholesky_solve(b[:, -1, :, None], chol_last).squeeze(-1)
            value = b / J.diagonal(dim1=-2, dim2=-1)
        else:
            chol = torch.linalg.cholesky(J)
            mean = torch.cholesky_solve(b.unsqueeze(-1), chol).squeeze(-1)
            chol_last, mean_last = chol[:, -1], mean[:, -1]
            value = b / J.diagonal(dim1=-2, dim2=-1) if mode == "F1" else mean
            if mode == "F3" or return_details:
                covariance = torch.cholesky_inverse(chol)
        uncertainty = (covariance.diagonal(dim1=-2, dim2=-1).clamp_min(1e-12).log()
                       if mode == "F3" else torch.zeros_like(value))
        features = torch.cat((value, uncertainty), -1)
        count = mask.sum((1, 2))
        logdet = 2 * chol_last.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        quadratic = (residual.square() / variance).sum((1, 2))
        log_noise = (mask * variance.log()).sum((1, 2))
        nll = .5 * (count * math.log(2 * math.pi) + log_noise + logdet + quadratic
                     - (b[:, -1] * mean_last).sum(-1)) / count.clamp_min(1)
        if return_details:
            return features, nll, {"precision": J, "information": b, "mean": mean, "covariance": covariance}
        return features, nll


class FusionS4(nn.Module):
    def __init__(self, variant: str, cfg: ModelConfig, seed: int):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant, self.cfg = variant, cfg
        # manual_seed also changes CUDA RNG; preserve it so extra frontend
        # initialization does not silently change the shared dropout stream.
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            self.blocks = nn.ModuleList([S4ResidualBlock(cfg) for _ in range(cfg.layers)])
            self.time_head = nn.Linear(cfg.seq_len, cfg.pred_len)
            self.channel_head = nn.Linear(cfg.width, cfg.channels)
            torch.manual_seed(seed + 2000)
            self.input_projection = nn.Linear(2 * cfg.channels, cfg.width)
            self.observation = self.encoding_projection = None
            if variant != "F0":
                torch.manual_seed(seed + 1000)
                self.observation = ConditionalEncoder(cfg)
                self.encoding_projection = nn.Linear(2 * cfg.rank, cfg.width, bias=False)
                nn.init.zeros_(self.encoding_projection.weight)

    def forward(self, x, mask):
        history = torch.cat((interpolate_history(x, mask), mask), -1)
        hidden = self.input_projection(history)
        nll = x.new_zeros(x.shape[0])
        if self.observation is not None:
            features, nll = self.observation(x, mask, mode=self.variant)
            hidden = hidden + self.encoding_projection(features)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.time_head(hidden.transpose(1, 2)).transpose(1, 2)
        return self.channel_head(hidden), nll

    def parameter_counts(self):
        return {"total": sum(p.numel() for p in self.parameters()),
                "history_projection": sum(p.numel() for p in self.input_projection.parameters()),
                "conditional_encoder": 0 if self.observation is None else sum(p.numel() for p in self.observation.parameters()),
                "encoding_projection": 0 if self.encoding_projection is None else sum(p.numel() for p in self.encoding_projection.parameters()),
                "s4_blocks": sum(p.numel() for p in self.blocks.parameters()),
                "head": sum(p.numel() for m in (self.time_head, self.channel_head) for p in m.parameters())}
