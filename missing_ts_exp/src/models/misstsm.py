"""MissTSM 紧凑实现：在 backbone 之前加入一个"缺失特征感知"的处理层。

核心思想：将每个 (t, c) 标量先嵌入到 q_dim 维向量，再用 mask 控制交叉注意力对缺失变量的可见性，
最终在每个时间步聚合得到一个跨变量表征，再投影回原维度，供下游预测 backbone 使用。

参考：A. Neog et al., arXiv:2502.15785, 2025.
原始代码：external/MissTSM/forecasting/misstsm_itransformer/layers/Transformer_EncDec.py
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn

from .dlinear import DLinear
from .patchtst import PatchTST
from .itransformer import iTransformer


class _LinearEmbed(nn.Module):
    def __init__(self, q_dim: int):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(1, q_dim), nn.LayerNorm(q_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C) -> (B, L, C, q_dim)
        return self.embed(x.unsqueeze(-1))


class _PE2D(nn.Module):
    """二维正弦位置编码（沿 L 和 C）。"""

    def __init__(self, q_dim: int):
        super().__init__()
        assert q_dim % 4 == 0, "q_dim must be divisible by 4 for 2D PE"
        self.q_dim = q_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C, D)
        B, L, C, D = x.shape
        half = D // 2
        # 沿 L
        pe_l = torch.zeros(L, half, device=x.device)
        pos_l = torch.arange(L, device=x.device).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, half, 2, device=x.device).float() * -(math.log(10000.0) / half))
        pe_l[:, 0::2] = torch.sin(pos_l * div)
        pe_l[:, 1::2] = torch.cos(pos_l * div)
        # 沿 C
        pe_c = torch.zeros(C, half, device=x.device)
        pos_c = torch.arange(C, device=x.device).float().unsqueeze(1)
        div2 = torch.exp(torch.arange(0, half, 2, device=x.device).float() * -(math.log(10000.0) / half))
        pe_c[:, 0::2] = torch.sin(pos_c * div2)
        pe_c[:, 1::2] = torch.cos(pos_c * div2)
        # 拼成 (L, C, D)
        pe = torch.cat([pe_l.unsqueeze(1).expand(L, C, half), pe_c.unsqueeze(0).expand(L, C, half)], dim=-1)
        return pe.unsqueeze(0).expand(B, L, C, D)


class MissTSMLayer(nn.Module):
    """MissTSM 核心：跨变量交叉注意力 + 缺失感知 padding mask。

    输入: x (B, L, C), mask (B, L, C) — 1=有观测，0=缺失
    输出: y (B, L, C_out)

    variant:
      - "full": 原始单 query
      - "cond_q": 时间步条件化 query (C1)
      - "multi_q": 多 query K=8 + mean pooling (C2)
      - "grouped_q": 将 C 通道分为 G 组，每组独立 query + cross-attention
    """

    def __init__(self, n_channels: int, q_dim: int = 64, num_heads: int = 4,
                 out_dim: int = None, variant: str = "full", n_queries: int = 8,
                 n_groups: int = 4):
        super().__init__()
        self.q_dim = q_dim
        self.n_channels = n_channels
        self.out_dim = out_dim if out_dim else n_channels
        self.variant = variant
        self.n_groups = n_groups

        if variant == "multi_q":
            self.var_query = nn.Parameter(torch.zeros(1, n_queries, q_dim))
        elif variant == "grouped_q":
            self.var_query = nn.Parameter(torch.zeros(1, n_groups, q_dim))
        else:
            self.var_query = nn.Parameter(torch.zeros(1, 1, q_dim))

        self.mask_embed = _LinearEmbed(q_dim)
        self.pos_embed = _PE2D(q_dim)
        if variant == "grouped_q":
            self.mhca_groups = nn.ModuleList([
                nn.MultiheadAttention(embed_dim=q_dim, num_heads=num_heads, batch_first=True)
                for _ in range(n_groups)
            ])
            self.group_proj = nn.Linear(n_groups * q_dim, q_dim)
        else:
            self.mhca = nn.MultiheadAttention(embed_dim=q_dim, num_heads=num_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(q_dim)
        self.projection = nn.Linear(q_dim, self.out_dim)
        nn.init.trunc_normal_(self.var_query, std=0.02)

        if variant == "cond_q":
            pe_dim = q_dim // 2
            self.time_proj = nn.Linear(pe_dim, q_dim)

    def _revin(self, x: torch.Tensor, mask: torch.Tensor):
        m_sum = mask.sum(dim=1).clamp(min=1.0)
        means = (x * mask).sum(dim=1) / m_sum
        means = means.unsqueeze(1)
        x = x - means
        var = ((x * mask) ** 2).sum(dim=1) / m_sum + 1e-5
        std = torch.sqrt(var).unsqueeze(1)
        x = x / std
        return x, means, std

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x_n, means, std = self._revin(x * mask, mask)
        emb = self.mask_embed(x_n)  # (B, L, C, q_dim)
        emb = emb + self.pos_embed(emb)
        B, L, C, D = emb.shape
        emb = emb.reshape(B * L, C, D)

        # Build query based on variant
        if self.variant == "cond_q":
            half = D // 2
            pos = torch.arange(L, device=x.device).float().unsqueeze(1)
            div = torch.exp(torch.arange(0, half, 2, device=x.device).float() * -(math.log(10000.0) / half))
            pe_time = torch.zeros(L, half, device=x.device)
            pe_time[:, 0::2] = torch.sin(pos * div)
            pe_time[:, 1::2] = torch.cos(pos * div)
            time_bias = self.time_proj(pe_time)  # (L, q_dim)
            q = self.var_query + time_bias.unsqueeze(0)  # (1, L, q_dim)
            q = q.expand(B, L, D).reshape(B * L, 1, D)
        elif self.variant == "multi_q":
            K = self.var_query.size(1)
            q = self.var_query.expand(B * L, K, D)
        elif self.variant != "grouped_q":
            q = self.var_query.expand(B * L, 1, D)

        pad_mask = (mask.reshape(B * L, C) < 0.5)
        all_missing = pad_mask.all(dim=1)
        if all_missing.any():
            pad_mask = pad_mask.clone()
            pad_mask[all_missing, 0] = False

        if self.variant == "grouped_q":
            G = self.n_groups
            group_size = (C + G - 1) // G
            group_outs = []
            for g in range(G):
                c_start = g * group_size
                c_end = min((g + 1) * group_size, C)
                emb_g = emb[:, c_start:c_end, :]
                pad_g = pad_mask[:, c_start:c_end]
                all_miss_g = pad_g.all(dim=1)
                if all_miss_g.any():
                    pad_g = pad_g.clone()
                    pad_g[all_miss_g, 0] = False
                q_g = self.var_query[:, g:g+1, :].expand(B * L, 1, D)
                ao, _ = self.mhca_groups[g](q_g, emb_g, emb_g, key_padding_mask=pad_g)
                group_outs.append(ao.squeeze(1))
            attn_out = self.group_proj(torch.cat(group_outs, dim=-1)).unsqueeze(1)
        else:
            attn_out, _ = self.mhca(q, emb, emb, key_padding_mask=pad_mask)

        if self.variant == "multi_q":
            attn_out = attn_out.mean(dim=1, keepdim=True)  # (B*L, 1, D)

        out = attn_out.reshape(B, L, D)
        out = self.layernorm(out)
        out = self.projection(out)  # (B, L, out_dim)
        if self.out_dim == self.n_channels:
            out = out * std + means
        return out


class MissTSMModel(nn.Module):
    """MissTSM 完整预测模型：缺失感知层 + 选定 backbone。"""

    name = "MissTSM"

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_channels: int,
        backbone: str = "iTransformer",
        q_dim: int = 64,
        num_heads: int = 4,
        d_model: int = 128,
        n_heads: int = 8,
        e_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        time_feat_dim: int = 0,
        patch_len: int = 16,
        stride: int = 8,
        skip: bool = True,
        variant: str = "full",
    ):
        super().__init__()
        self.skip = skip
        self.variant = variant

        mtsm_variant = variant if variant in ("cond_q", "multi_q", "grouped_q") else "full"
        n_groups = 4
        if variant.startswith("grouped_q"):
            mtsm_variant = "grouped_q"
            parts = variant.split("_q")
            if len(parts) == 2 and parts[1].isdigit():
                n_groups = int(parts[1])
        self.mtsm = MissTSMLayer(n_channels, q_dim=q_dim, num_heads=num_heads,
                                 out_dim=n_channels, variant=mtsm_variant,
                                 n_groups=n_groups)

        if variant == "soft_skip":
            self.skip_gate = nn.Linear(n_channels, n_channels)
        backbone = backbone.lower()
        if backbone == "itransformer":
            self.backbone = iTransformer(
                seq_len, pred_len, n_channels,
                d_model=d_model, n_heads=n_heads, e_layers=e_layers,
                d_ff=d_ff, dropout=dropout, time_feat_dim=time_feat_dim, use_norm=False,
            )
        elif backbone == "patchtst":
            self.backbone = PatchTST(
                seq_len, pred_len, n_channels,
                patch_len=patch_len, stride=stride, d_model=d_model,
                n_heads=n_heads, e_layers=e_layers, d_ff=d_ff, dropout=dropout,
                use_revin=False,
            )
        elif backbone == "dlinear":
            self.backbone = DLinear(seq_len, pred_len, n_channels)
        else:
            raise ValueError(backbone)

    def forward(self, x: torch.Tensor, x_mark=None, mask=None) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(x)
        feat = self.mtsm(x, mask)  # (B, L, C)
        if self.variant == "soft_skip":
            alpha = torch.sigmoid(self.skip_gate(feat))
            feat = mask * (alpha * x + (1 - alpha) * feat) + (1 - mask) * feat
        elif self.skip:
            feat = mask * x + (1 - mask) * feat
        return self.backbone(feat, x_mark)
