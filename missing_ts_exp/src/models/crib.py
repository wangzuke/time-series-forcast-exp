"""CRIB: Contrastive Representation learning with Information Bottleneck for time series forecasting
under missing observations. Full implementation based on the original paper repo
(github.com/Muyiiiii/CRIB).

Key ideas:
- Patch-based TCN + Transformer encoder produces per-channel latent distributions (loc, scale)
  forming a variational information bottleneck regularised against N(0, I).
- Dual-view contrastive consistency (small Gaussian perturbation) encourages robust representations.
- Mask-aware RevIN normalises using only observed (mask=1) positions.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.distributions import MultivariateNormal, kl_divergence


# ---------------------------------------------------------------------------
# Bottom-level modules
# ---------------------------------------------------------------------------

class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., :-self.chomp_size]


class TemporalBlock(nn.Module):
    """Dilated causal Conv2d block operating on (B, in_ch, C, P) tensors."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int,
                 dilation: int, padding: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = weight_norm(
            nn.Conv2d(in_ch, out_ch, (1, kernel_size),
                      stride=(1, stride), padding=(0, padding), dilation=(1, dilation))
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv2d(out_ch, out_ch, (1, kernel_size),
                      stride=(1, stride), padding=(0, padding), dilation=(1, dilation))
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv2d(in_ch, out_ch, (1, 1)) if in_ch != out_ch else None
        )
        self.relu_out = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.drop1(self.relu1(self.chomp1(self.conv1(x))))
        out = self.drop2(self.relu2(self.chomp2(self.conv2(out))))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu_out(out + res)


class TCNBlock(nn.Module):
    """Stack of TemporalBlocks with exponentially growing dilation."""

    def __init__(self, in_channel: int, out_channel_list: list[int],
                 kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        layers = []
        in_ch = in_channel
        for i, out_ch in enumerate(out_channel_list):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, stride=1,
                                        dilation=dilation, padding=padding,
                                        dropout=dropout))
            in_ch = out_ch
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq, d_model)
        return x + self.pe[:, : x.size(1)]


# ---------------------------------------------------------------------------
# Transformer modules
# ---------------------------------------------------------------------------

class CRIBAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        H, d_k = self.n_heads, self.d_k

        Q = self.W_q(x).view(B, N, H, d_k).transpose(1, 2)
        K = self.W_k(x).view(B, N, H, d_k).transpose(1, 2)
        V = self.W_v(x).view(B, N, H, d_k).transpose(1, 2)

        attn = torch.softmax(torch.matmul(Q, K.transpose(-2, -1)) / self.scale, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)  # (B, H, N, d_k)
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.out_proj(out)


class CRIBEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = CRIBAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Conv1d(d_model, d_ff * 4, 1),
            nn.GELU(),
            nn.Conv1d(d_ff * 4, d_model, 1),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.drop(self.attn(x)))
        ff_out = self.ff(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm2(x + self.drop(ff_out))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, layers: list[nn.Module], norm: nn.Module | None = None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class CRIBEncoder(nn.Module):
    def __init__(self, patch_len: int, patch_num: int, n_channels: int,
                 d_model: int, n_heads: int, e_layers: int, d_ff: int, dropout: float):
        super().__init__()
        self.patch_num = patch_num
        self.n_channels = n_channels
        self.d_model = d_model

        self.tcn = TCNBlock(
            in_channel=patch_len,
            out_channel_list=[64, d_model],
            kernel_size=3,
            dropout=dropout,
        )
        self.transformer = TransformerEncoder(
            [CRIBEncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(e_layers)]
        )
        self.projector = nn.Sequential(
            nn.Linear(patch_num * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model * 2),
        )

    def forward(self, x: torch.Tensor):
        # x: (B, P, C, patch_len)
        B, P, C, PL = x.shape

        # TCN expects (B, in_ch=patch_len, C, P)
        tcn_in = x.permute(0, 3, 2, 1)             # (B, patch_len, C, P)
        tcn_out = self.tcn(tcn_in)                  # (B, d_model, C, P)

        # Transformer on (B, P*C, d_model)
        tr_in = tcn_out.permute(0, 3, 2, 1)         # (B, P, C, d_model)
        tr_in = tr_in.reshape(B, P * C, self.d_model)
        enc_out = self.transformer(tr_in)            # (B, P*C, d_model)

        # Build variational distribution per channel
        # enc_out tokens are in (P, C) order: reshape as (B, P, C, d_model) then transpose
        h = enc_out.reshape(B, P, C, self.d_model).permute(0, 2, 1, 3)  # (B, C, P, d_model)
        h = h.reshape(B, C, P * self.d_model)
        mapped = self.projector(h)                   # (B, C, d_model*2)

        loc = mapped[:, :, : self.d_model]
        scale = F.softplus(mapped[:, :, self.d_model:]) + 1e-9
        qz = MultivariateNormal(loc=loc, covariance_matrix=torch.diag_embed(scale))

        return enc_out, qz


# ---------------------------------------------------------------------------
# Prediction head
# ---------------------------------------------------------------------------

class CRIBPredHead(nn.Module):
    def __init__(self, patch_num: int, n_channels: int, d_model: int, pred_len: int):
        super().__init__()
        self.patch_num = patch_num
        self.n_channels = n_channels
        self.d_model = d_model
        self.pred1 = nn.Linear(patch_num * d_model, d_model)
        self.pred2 = nn.Linear(d_model, pred_len)

    def forward(self, enc_out: torch.Tensor) -> torch.Tensor:
        # enc_out: (B, P*C, d_model), tokens in (P, C) order
        B = enc_out.shape[0]
        h = enc_out.reshape(B, self.patch_num, self.n_channels, self.d_model)
        h = h.permute(0, 2, 1, 3)                          # (B, C, P, d_model)
        h = h.reshape(B, self.n_channels, self.patch_num * self.d_model)
        h = F.relu(self.pred1(h))                   # (B, C, d_model)
        h = self.pred2(h)                            # (B, C, pred_len)
        return h.permute(0, 2, 1)                   # (B, pred_len, C)


# ---------------------------------------------------------------------------
# CRIB main class
# ---------------------------------------------------------------------------

class CRIB(nn.Module):
    name = "CRIB"

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_channels: int,
        d_model: int = 128,
        n_heads: int = 8,
        e_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        kl_weight: float = 1e-3,
        sample_train: bool = True,
        use_mask: bool = True,
        patch_len: int = 12,
        consistency_weight: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_channels = n_channels
        self.d_model = d_model
        self.kl_weight = kl_weight
        self.sample_train = sample_train
        self.consistency_weight = consistency_weight
        self.patch_len = patch_len
        self.eps = 1e-5

        # Pad seq_len up to a multiple of patch_len
        pad = (patch_len - seq_len % patch_len) % patch_len
        self.pad_len = pad
        L_padded = seq_len + pad
        self.patch_num = L_padded // patch_len

        self.pos_embed = PositionalEmbedding(patch_len)

        self.encoder = CRIBEncoder(
            patch_len=patch_len,
            patch_num=self.patch_num,
            n_channels=n_channels,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.pred_head = CRIBPredHead(
            patch_num=self.patch_num,
            n_channels=n_channels,
            d_model=d_model,
            pred_len=pred_len,
        )

        self._last_kl = torch.tensor(0.0)
        self._last_consistency = torch.tensor(0.0)
        # Lazily built prior cached to avoid repeated allocation
        self._prior: MultivariateNormal | None = None

    def _get_prior(self, B: int, C: int, device: torch.device) -> MultivariateNormal:
        # Rebuild only when device or shape changes (rare)
        if (self._prior is None
                or self._prior.loc.device != device
                or self._prior.loc.shape != (B, C, self.d_model)):
            loc = torch.zeros(B, C, self.d_model, device=device)
            cov = torch.eye(self.d_model, device=device).unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
            self._prior = MultivariateNormal(loc=loc, covariance_matrix=cov)
        return self._prior

    def _revin_norm(self, x: torch.Tensor, mask: torch.Tensor):
        # mask: (B, L, C), 1 = observed
        m_sum = mask.sum(dim=1).clamp(min=1.0)          # (B, C)
        mean = (x * mask).sum(dim=1) / m_sum             # (B, C)
        x_c = x - mean.unsqueeze(1)
        var = ((x_c * mask) ** 2).sum(dim=1) / m_sum + self.eps
        std = var.sqrt()
        x_n = x_c / std.unsqueeze(1)
        return x_n, mean, std

    def _revin_denorm(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        # x: (B, H, C), mean/std: (B, C)
        return x * std.unsqueeze(1) + mean.unsqueeze(1)

    def _to_patches(self, x: torch.Tensor, mask: torch.Tensor):
        """Zero-pad then unfold into (B, P, C, patch_len)."""
        B, L, C = x.shape
        if self.pad_len > 0:
            x = F.pad(x, (0, 0, 0, self.pad_len))       # pad time dim
            mask = F.pad(mask, (0, 0, 0, self.pad_len))  # extend mask with zeros
        # x: (B, L_padded, C) -> (B, C, L_padded) -> unfold
        x = x.permute(0, 2, 1)                           # (B, C, L_padded)
        x = x.unfold(2, self.patch_len, self.patch_len)  # (B, C, P, patch_len)
        x = x.permute(0, 2, 1, 3)                        # (B, P, C, patch_len)
        return x

    def forward(self, x: torch.Tensor, x_mark=None, mask: torch.Tensor | None = None):
        if mask is None:
            mask = torch.ones_like(x)

        # Zero out missing positions before any computation
        x = x * mask

        # Mask-aware RevIN
        x_n, mean, std = self._revin_norm(x, mask)
        x_n = x_n * mask                              # keep missing positions at 0

        # Patch
        x1 = self._to_patches(x_n, mask)             # (B, P, C, patch_len)

        # Add sinusoidal positional bias over the patch_len axis (within-patch positions).
        # self.pos_embed.pe: (1, max_len, patch_len); pe[0, i, i] gives the i-th diagonal
        # element. We use pe[0, :PL, 0] — one scalar per position — broadcast over (B,P,C).
        B, P, C, PL = x1.shape
        # Take the first feature column of the sinusoidal table as a position scalar
        pos_bias = self.pos_embed.pe[0, :PL, 0]        # (PL,)
        x1 = x1 + pos_bias.view(1, 1, 1, PL)          # broadcast to (B, P, C, PL)

        if self.training:
            x2 = x1 + 0.01 * torch.randn_like(x1)
            enc_out_1, qz_1 = self.encoder(x1)
            enc_out_2, qz_2 = self.encoder(x2)

            # KL against standard normal prior
            prior = self._get_prior(B, C, x.device)
            kl_raw = kl_divergence(qz_1, prior)            # (B, C)
            kl_raw = torch.nan_to_num(kl_raw, nan=0.0, posinf=0.0, neginf=0.0)
            kl = kl_raw.mean()

            # Symmetric consistency loss
            cons = (
                F.mse_loss(enc_out_1.detach(), enc_out_2)
                + F.mse_loss(enc_out_1, enc_out_2.detach())
            ) / 2.0

            self._last_kl = kl
            self._last_consistency = cons

            z = qz_1.rsample()                             # (B, C, d_model)
        else:
            enc_out_1, qz_1 = self.encoder(x1)

            prior = self._get_prior(B, C, x.device)
            kl_raw = kl_divergence(qz_1, prior)            # (B, C)
            kl_raw = torch.nan_to_num(kl_raw, nan=0.0, posinf=0.0, neginf=0.0)
            self._last_kl = kl_raw.mean()
            self._last_consistency = torch.tensor(0.0, device=x.device)

            z = qz_1.mean

        # Prediction from encoder output (not from sampled z directly, as in original)
        pred = self.pred_head(enc_out_1)               # (B, pred_len, C)

        # RevIN denorm
        pred = self._revin_denorm(pred, mean, std)
        return pred

    def auxiliary_loss(self) -> torch.Tensor:
        kl = self._last_kl
        cons = self._last_consistency
        if not torch.is_tensor(kl):
            kl = torch.tensor(float(kl))
        if not torch.is_tensor(cons):
            cons = torch.tensor(float(cons))
        return self.kl_weight * kl + self.consistency_weight * cons
