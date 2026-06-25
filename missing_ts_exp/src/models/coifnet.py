"""CoIFNet: Joint Imputation-Forecasting Network.

Full implementation based on: K. Tang et al., arXiv:2506.13064, 2025.
Reference repo: github.com/KaiTang-eng/CoIFNet

Architecture: dual-pathway SharedModule (intra=temporal, inter=channel) with
mask-aware RevON normalization. A single backbone outputs both history imputation
and future forecast simultaneously.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class TSBlock(nn.Module):
    """Token-mixing MLP with GEGLU gating."""

    def __init__(self, input_dim: int, output_dim: int, mid_hidden: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, mid_hidden * 2)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(mid_hidden, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x, gate = x.chunk(2, dim=-1)
        x = x * F.gelu(gate)
        x = self.drop(x)
        return self.fc2(x)


class AttentionBlock(nn.Module):
    """Multi-head attention with sigmoid-gated values."""

    def __init__(self, dim_in: int, dim_out: int, dim_head: int = 32, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim_in, inner_dim * 3, bias=False)
        self.to_v_gates = nn.Linear(dim_in, heads)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim_out),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, dim_in)
        B, N, _ = x.shape
        h, d = self.heads, self.dim_head

        qkv = self.to_qkv(x)  # (B, N, 3*h*d)
        q, k, v = qkv.chunk(3, dim=-1)  # each (B, N, h*d)

        # reshape to (B, h, N, d)
        q = q.reshape(B, N, h, d).permute(0, 2, 1, 3)
        k = k.reshape(B, N, h, d).permute(0, 2, 1, 3)
        v = v.reshape(B, N, h, d).permute(0, 2, 1, 3)

        # gates: (B, N, h) -> sigmoid -> (B, h, N, 1)
        gates = torch.sigmoid(self.to_v_gates(x))  # (B, N, h)
        gates = gates.permute(0, 2, 1).unsqueeze(-1)  # (B, h, N, 1)

        # scaled dot-product attention in float32
        attn = torch.matmul(q.float(), k.float().transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1).to(v.dtype)

        out = torch.matmul(attn, v)  # (B, h, N, d)
        out = out * gates

        # merge heads: (B, h, N, d) -> (B, N, h*d)
        out = out.permute(0, 2, 1, 3).reshape(B, N, h * d)
        return self.to_out(out)


def _make_block(block_type: str, input_dim: int, output_dim: int,
                heads: int, dropout: float) -> nn.Module:
    if block_type == "TSBlock":
        mid = max(input_dim, output_dim)
        return TSBlock(input_dim, output_dim, mid, dropout)
    elif block_type == "AttentionBlock":
        return AttentionBlock(input_dim, output_dim, heads=heads, dropout=dropout)
    else:
        raise ValueError(f"Unknown block type: {block_type}")


# ---------------------------------------------------------------------------
# Mask-aware normalization
# ---------------------------------------------------------------------------

class RevON(nn.Module):
    """3-mode mask-aware reversible normalization.

    Modes:
        "norm"      — mask-aware stats, normalize, zero out missing positions
        "norm-fore" — normalize with stored stats (no recompute)
        "denorm"    — reverse normalization with stored stats
    """

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        self.mean_: torch.Tensor
        self.std_: torch.Tensor

    def forward(self, x: torch.Tensor, mode: str, mask: torch.Tensor = None) -> torch.Tensor:
        if mode == "norm":
            if mask is not None:
                m_sum = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
                mean = (x * mask).sum(dim=1, keepdim=True) / m_sum
                var = ((x - mean).pow(2) * mask).sum(dim=1, keepdim=True) / m_sum
            else:
                mean = x.mean(dim=1, keepdim=True)
                var = x.var(dim=1, keepdim=True, unbiased=False)
            std = (var + self.eps).sqrt()
            self.mean_ = mean
            self.std_ = std
            x_normed = (x - mean) / std
            if self.affine:
                x_normed = x_normed * self.weight + self.bias
            if mask is not None:
                x_normed = x_normed * mask
            return x_normed

        elif mode == "norm-fore":
            x_normed = (x - self.mean_) / self.std_
            if self.affine:
                x_normed = x_normed * self.weight + self.bias
            return x_normed

        elif mode == "denorm":
            if self.affine:
                x = (x - self.bias) / (self.weight + self.eps)
            return x * self.std_ + self.mean_

        else:
            raise ValueError(f"Unknown RevON mode: {mode}")


# ---------------------------------------------------------------------------
# Shared dual-pathway module
# ---------------------------------------------------------------------------

class _SharedLayer(nn.Module):
    """One layer of the shared module: temporal (intra) + channel (inter) mixing."""

    def __init__(self, total_len: int, in_channel_dim: int, out_channel_dim: int,
                 intra_type: str, inter_type: str, heads: int, dropout: float):
        super().__init__()
        self.intra_norm = nn.LayerNorm(total_len)
        self.inter_norm = nn.LayerNorm(in_channel_dim)

        self.intra_model = _make_block(intra_type, total_len, total_len, heads, dropout)
        self.inter_model = _make_block(inter_type, in_channel_dim, out_channel_dim, heads, dropout)

        self.same_channel_dim = (in_channel_dim == out_channel_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, total_len, C)
        # --- temporal mixing ---
        x_t = x.permute(0, 2, 1)          # (B, C, total_len)
        x_t = self.intra_norm(x_t)
        x_t = self.intra_model(x_t)        # (B, C, total_len)
        x_t = x_t.permute(0, 2, 1)         # (B, total_len, C)
        if self.same_channel_dim:
            x = x + x_t
        else:
            x = x_t

        # --- channel mixing ---
        x_c = self.inter_norm(x)
        x_c = self.inter_model(x_c)
        if self.same_channel_dim:
            x = x + x_c
        else:
            x = x_c

        return x


class SharedModule(nn.Module):
    def __init__(self, total_len: int, in_channel_dim: int, out_channel_dim: int,
                 n_layers: int, intra_type: str, inter_type: str,
                 heads: int, dropout: float):
        super().__init__()
        layers = []
        for i in range(n_layers):
            in_dim = in_channel_dim if i == 0 else out_channel_dim
            layers.append(_SharedLayer(
                total_len, in_dim, out_channel_dim,
                intra_type, inter_type, heads, dropout,
            ))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# ---------------------------------------------------------------------------
# CoIFNet
# ---------------------------------------------------------------------------

class CoIFNet(nn.Module):
    """Joint imputation-forecasting network.

    Inputs : x     (B, L, C)  — observed values (0 at missing positions)
             x_mark (B, L, F) — optional time features (history segment only)
             mask  (B, L, C)  — 1=observed, 0=missing
    Outputs: {"forecast": (B, H, C), "impute": (B, L, C)}
    """

    name = "CoIFNet"

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_channels: int,
        hidden: int = 128,
        n_layers: int = 3,
        dropout: float = 0.1,
        use_revin: bool = True,
        impute_weight: float = 0.5,
        intra_type: str = "TSBlock",
        inter_type: str = "AttentionBlock",
        n_heads: int = 4,
        use_time_feat: bool = True,
        time_feat_proj: int = 8,
        time_feat_dim: int = 4,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.total_len = seq_len + pred_len
        self.n_channels = n_channels
        self.hidden = hidden
        self.impute_weight = impute_weight
        self.use_revin = use_revin
        self.use_time_feat = use_time_feat
        self.time_feat_proj = time_feat_proj

        if use_revin:
            self.revin = RevON(n_channels)

        in_dim = n_channels * 2

        # embedding + positional
        embed_in_dim = in_dim + (time_feat_proj if use_time_feat else 0)
        self.embed = nn.Linear(embed_in_dim, hidden)
        self.pos = nn.Parameter(torch.randn(1, self.total_len, hidden) * 0.02)

        if use_time_feat:
            self.time_proj = nn.Linear(time_feat_dim, time_feat_proj)

        self.backbone = SharedModule(
            total_len=self.total_len,
            in_channel_dim=hidden,
            out_channel_dim=hidden,
            n_layers=n_layers,
            intra_type=intra_type,
            inter_type=inter_type,
            heads=n_heads,
            dropout=dropout,
        )
        self.norm_out = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, n_channels)

    def forward(self, x: torch.Tensor, x_mark=None, mask: torch.Tensor = None):
        if mask is None:
            mask = torch.ones_like(x)

        B, L, C = x.shape
        x_orig = x

        # 1. Mask-aware normalization (history segment only)
        if self.use_revin:
            x_normed = self.revin(x, mode="norm", mask=mask)
        else:
            x_normed = x

        # 2. Build full-length input: [obs*mask | mask] for history, zeros for forecast
        pad_x = torch.zeros(B, self.pred_len, C, device=x.device, dtype=x.dtype)
        pad_m = torch.zeros(B, self.pred_len, C, device=x.device, dtype=x.dtype)

        x_full = torch.cat([x_normed * mask, pad_x], dim=1)  # (B, total, C)
        m_full = torch.cat([mask, pad_m], dim=1)              # (B, total, C)
        inp = torch.cat([x_full, m_full], dim=-1)             # (B, total, 2C)

        # 3. Optional time features
        if self.use_time_feat and x_mark is not None:
            t_hist = self.time_proj(x_mark)                     # (B, L, time_feat_proj)
            t_pad = torch.zeros(B, self.pred_len, self.time_feat_proj,
                                device=x.device, dtype=x.dtype)
            t_full = torch.cat([t_hist, t_pad], dim=1)         # (B, total, time_feat_proj)
            inp = torch.cat([inp, t_full], dim=-1)              # (B, total, 2C+tfp)

        # 4. Embed + positional
        h = self.embed(inp) + self.pos                           # (B, total, hidden)

        # 5. Backbone
        h = self.backbone(h)

        # 6. Output head
        h = self.norm_out(h)
        out_full = self.head(h)                               # (B, total, C)

        # 7. Denormalize
        if self.use_revin:
            out_full = self.revin(out_full, mode="denorm")

        # 8. Split history / forecast
        impute_raw = out_full[:, :self.seq_len, :]
        forecast = out_full[:, self.seq_len:, :]

        # 9. Preserve observed values in impute
        impute = mask * x_orig + (1.0 - mask) * impute_raw

        return {"forecast": forecast, "impute": impute}

    def compute_loss(
        self,
        out: dict,
        x_true: torch.Tensor,
        mask: torch.Tensor,
        y_true: torch.Tensor,
        criterion=None,
    ):
        """Compute combined forecast + imputation loss (backward compatibility)."""
        if criterion is None:
            criterion = nn.L1Loss(reduction="none")
        impute = out["impute"]
        forecast = out["forecast"]
        rec = criterion(impute, x_true) * mask
        rec_loss = rec.sum() / mask.sum().clamp(min=1.0)
        fc_loss = criterion(forecast, y_true).mean()
        return fc_loss + self.impute_weight * rec_loss, rec_loss.detach(), fc_loss.detach()
