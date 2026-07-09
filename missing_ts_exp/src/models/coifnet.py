"""CoIFNet: Joint Imputation-Forecasting Network.

Full implementation based on: K. Tang et al., arXiv:2506.13064, 2025.
Reference repo: github.com/KaiTang-eng/CoIFNet

Architecture: single-layer dual-pathway SharedModule (intra=temporal, inter=channel)
with mask-aware RevON normalization. The inter_model handles dimension compression
from 2C+feat_dim to C using GEGLU-gated TSBlock (hidden=256).

Original training flow (CoIFNetTask.py):
  - Input: x (B, seq_len, C), mask (B, seq_len, C)
  - SharedModule: intra_model maps seq_len→hidden, inter_model maps 2C+feat→C
  - aux_head: Linear(hidden, seq_len+pred_len) → output (B, seq_len+pred_len, C)
  - Split output[:, :seq_len] = imputation, output[:, seq_len:] = forecast
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class GEGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


class TSBlock(nn.Module):
    """Token-mixing MLP with GEGLU gating (matches original CoIFNet repo)."""

    def __init__(self, input_dim: int, output_dim: int, mid_hidden: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, mid_hidden * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(mid_hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


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
        B, N, _ = x.shape
        h, d = self.heads, self.dim_head

        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.reshape(B, N, h, d).permute(0, 2, 1, 3)
        k = k.reshape(B, N, h, d).permute(0, 2, 1, 3)
        v = v.reshape(B, N, h, d).permute(0, 2, 1, 3)

        gates = torch.sigmoid(self.to_v_gates(x))
        gates = gates.permute(0, 2, 1).unsqueeze(-1)

        attn = torch.matmul(q.float(), k.float().transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1).to(v.dtype)

        out = torch.matmul(attn, v)
        out = out * gates

        out = out.permute(0, 2, 1, 3).reshape(B, N, h * d)
        return self.to_out(out)


def _make_block(block_type: str, input_dim: int, output_dim: int,
                mid_hidden: int, dropout: float) -> nn.Module:
    if block_type == "TSBlock":
        return TSBlock(input_dim, output_dim, mid_hidden, dropout)
    elif block_type == "AttentionBlock":
        return AttentionBlock(input_dim, output_dim, dropout=dropout)
    else:
        raise ValueError(f"Unknown block type: {block_type}")


# ---------------------------------------------------------------------------
# Mask-aware normalization
# ---------------------------------------------------------------------------

class RevON(nn.Module):
    """3-mode mask-aware reversible normalization."""

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

        elif mode == "denorm":
            if self.affine:
                x = (x - self.bias) / (self.weight + self.eps)
            return x * self.std_ + self.mean_

        else:
            raise ValueError(f"Unknown RevON mode: {mode}")


# ---------------------------------------------------------------------------
# Shared dual-pathway module (single layer, matching original repo)
# ---------------------------------------------------------------------------

class SharedModule(nn.Module):
    """Single-layer dual-pathway: temporal (intra) + channel (inter) mixing.

    Matches the original CoIFNet repo architecture exactly:
    - intra_model: temporal mixing (in_seq → out_seq) operating on channel dim
    - inter_model: channel mixing (2C+feat_dim → C) using GEGLU compression
    """

    def __init__(self, in_seq: int, out_seq: int, n_channels: int,
                 in_channel_dim: int, hidden: int,
                 intra_type: str, inter_type: str, dropout: float):
        super().__init__()
        self.intra_model = _make_block(intra_type, in_seq, out_seq, hidden, dropout)
        self.inter_model = _make_block(inter_type, in_channel_dim, n_channels, hidden, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_seq, in_channel_dim)
        # temporal mixing: (B, in_channel_dim, in_seq) → (B, in_channel_dim, out_seq)
        x = self.intra_model(x.permute(0, 2, 1)).permute(0, 2, 1)
        # channel mixing: (B, out_seq, in_channel_dim) → (B, out_seq, C)
        x = self.inter_model(x)
        return x


# ---------------------------------------------------------------------------
# CoIFNet
# ---------------------------------------------------------------------------

class CoIFNet(nn.Module):
    """Joint imputation-forecasting network.

    Matches the original repo (github.com/KaiTang-eng/CoIFNet) architecture:
    - Single SharedModule: intra maps seq_len→hidden, inter maps 2C+feat→C
    - aux_head: Linear(hidden, seq_len+pred_len) maps to full horizon
    - Output split: [:seq_len] = imputation, [seq_len:] = forecast
    - Input: cat([x, mask], dim=-1), NO x*mask
    - hidden=256 (original config), inter_type=TSBlock (original config)

    Inputs : x     (B, L, C)  — observed values (0 at missing positions)
             x_mark (B, L, F) — optional time features
             mask  (B, L, C)  — 1=observed, 0=missing
    Outputs: {"forecast": (B, H, C), "impute": (B, L, C)}
    """

    name = "CoIFNet"

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_channels: int,
        hidden: int = 256,
        n_layers: int = 3,
        dropout: float = 0.1,
        use_revin: bool = True,
        impute_weight: float = 0.5,
        intra_type: str = "TSBlock",
        inter_type: str = "TSBlock",
        n_heads: int = 4,
        use_time_feat: bool = True,
        time_feat_proj: int = 8,
        time_feat_dim: int = 4,
        input_form: str = "x_cat_mask",
        embed_type: str = "shared",
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_channels = n_channels
        self.hidden = hidden
        self.impute_weight = impute_weight
        self.use_revin = use_revin
        self.use_time_feat = use_time_feat
        self.time_feat_proj = time_feat_proj
        self.horizon_len = seq_len + pred_len
        self.input_form = input_form
        self.embed_type = embed_type

        if use_revin:
            self.revin = RevON(n_channels)

        in_channel_dim = n_channels * 2
        if use_time_feat:
            in_channel_dim += time_feat_proj
            self.time_proj = nn.Linear(time_feat_dim, time_feat_proj)

        if embed_type == "independent":
            self.embed_proj = nn.Linear(in_channel_dim, n_channels)
            self.intra_model = _make_block(intra_type, seq_len, hidden, hidden, dropout)
        else:
            self.shared_model = SharedModule(
                in_seq=seq_len,
                out_seq=hidden,
                n_channels=n_channels,
                in_channel_dim=in_channel_dim,
                hidden=hidden,
                intra_type=intra_type,
                inter_type=inter_type,
                dropout=dropout,
            )
        self.aux_head = nn.Linear(hidden, self.horizon_len)

    def forward(self, x: torch.Tensor, x_mark=None, mask: torch.Tensor = None):
        if mask is None:
            mask = torch.ones_like(x)

        B, L, C = x.shape
        x_orig = x

        # 1. Mask-aware RevON normalization
        if self.use_revin:
            x_normed = self.revin(x, mode="norm", mask=mask)
        else:
            x_normed = x

        # 2. Build input
        if self.input_form == "xmask_cat_mask":
            inp = torch.cat([x_normed * mask, mask], dim=-1)  # (B, L, 2C)
        else:
            inp = torch.cat([x_normed, mask], dim=-1)  # (B, L, 2C)

        # 3. Optional time features (zero-pad if x_mark unavailable)
        if self.use_time_feat:
            if x_mark is not None:
                t_feat = self.time_proj(x_mark)
            else:
                t_feat = torch.zeros(B, L, self.time_feat_proj, device=x.device, dtype=x.dtype)
            inp = torch.cat([inp, t_feat], dim=-1)  # (B, L, 2C+tfp)

        # 4. Encode: SharedModule or independent embedding path
        if self.embed_type == "independent":
            h = self.embed_proj(inp)  # (B, seq_len, C)
            h = self.intra_model(h.permute(0, 2, 1)).permute(0, 2, 1)  # (B, hidden, C)
        else:
            h = self.shared_model(inp)

        # 5. aux_head: (B, C, hidden) → (B, C, horizon_len) → (B, horizon_len, C)
        out_full = self.aux_head(h.permute(0, 2, 1)).permute(0, 2, 1)

        # 6. Denormalize
        if self.use_revin:
            out_full = self.revin(out_full, mode="denorm")

        # 7. Split: imputation (first seq_len) and forecast (last pred_len)
        impute_raw = out_full[:, :self.seq_len, :]
        forecast = out_full[:, self.seq_len:, :]

        # 8. Preserve observed values in imputation
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
        if criterion is None:
            criterion = nn.L1Loss(reduction="none")
        impute = out["impute"]
        forecast = out["forecast"]
        rec = criterion(impute, x_true) * mask
        rec_loss = rec.sum() / mask.sum().clamp(min=1.0)
        fc_loss = criterion(forecast, y_true).mean()
        return fc_loss + self.impute_weight * rec_loss, rec_loss.detach(), fc_loss.detach()
