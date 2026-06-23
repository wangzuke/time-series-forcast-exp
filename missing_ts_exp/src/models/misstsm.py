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
    """

    def __init__(self, n_channels: int, q_dim: int = 64, num_heads: int = 4, out_dim: int = None):
        super().__init__()
        self.q_dim = q_dim
        self.n_channels = n_channels
        self.out_dim = out_dim if out_dim else n_channels
        self.var_query = nn.Parameter(torch.zeros(1, 1, q_dim))
        self.mask_embed = _LinearEmbed(q_dim)
        self.pos_embed = _PE2D(q_dim)
        self.mhca = nn.MultiheadAttention(embed_dim=q_dim, num_heads=num_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(q_dim)
        self.projection = nn.Linear(q_dim, self.out_dim)
        nn.init.trunc_normal_(self.var_query, std=0.02)

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
        # 用 mask 做 RevIN（仅在有观测的位置计算统计量）
        x_n, means, std = self._revin(x * mask, mask)
        # 嵌入每个标量
        emb = self.mask_embed(x_n)  # (B, L, C, q_dim)
        emb = emb + self.pos_embed(emb)
        B, L, C, D = emb.shape
        emb = emb.reshape(B * L, C, D)
        # query (B*L, 1, D)
        q = self.var_query.expand(B * L, 1, D)
        # key_padding_mask: True 表示该位置应被忽略；mask=0 -> 缺失 -> True
        pad_mask = (mask.reshape(B * L, C) < 0.5)
        # 防止某个时间步全是缺失，attention 会 NaN：给至少一个位置允许参与（用 var_query 自身做 self-attend）
        all_missing = pad_mask.all(dim=1)
        if all_missing.any():
            pad_mask = pad_mask.clone()
            pad_mask[all_missing, 0] = False
        attn_out, _ = self.mhca(q, emb, emb, key_padding_mask=pad_mask)
        out = attn_out.reshape(B, L, D)
        out = self.layernorm(out)
        out = self.projection(out)  # (B, L, out_dim)
        # 反 RevIN（按 out_dim==n_channels 时简单还原）
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
    ):
        super().__init__()
        self.mtsm = MissTSMLayer(n_channels, q_dim=q_dim, num_heads=num_heads, out_dim=n_channels)
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
        return self.backbone(feat, x_mark)
