"""CRIB（紧凑版）：从带缺失的部分观测序列直接预测，使用信息瓶颈正则。

参考: J. Yang et al., arXiv:2509.23494, 2025.
原始代码: external/CRIB/TSL_models/CRIB.py

简化点：
- 保留核心思想：编码器输出隐表征的均值/方差 (μ, σ)，对 N(0, I) 做 KL 约束作为信息瓶颈；
  从隐表征采样后送入预测头。
- 输入显式拼接 mask 作为额外通道，使模型知道哪些位置是真实观测；不补值。
- 训练时同时计算预测 MSE + β * KL；β 通过 forward 时返回 KL 给训练循环来加权。
- 不实现原文的 patching + 两次 noise 采样的 ELBO；用单次重参数化采样，效果作为基线对比。
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn

from .itransformer import _EncoderLayer  # 复用 attention 编码器


class _CRIB_Embed(nn.Module):
    def __init__(self, n_channels: int, d_model: int, use_mask: bool = True):
        super().__init__()
        in_dim = n_channels * 2 if use_mask else n_channels
        self.use_mask = use_mask
        self.token = nn.Linear(in_dim, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.use_mask:
            x = torch.cat([x * mask, mask], dim=-1)
        else:
            x = x * mask
        return self.token(x)


class _PosEmbed(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


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
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_channels = n_channels
        self.kl_weight = kl_weight
        self.sample_train = sample_train
        self.embed = _CRIB_Embed(n_channels, d_model, use_mask=use_mask)
        self.pos = _PosEmbed(d_model)
        self.encoder = nn.ModuleList(
            [_EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(e_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        # 信息瓶颈头：每个时间步独立产生 μ, log σ²
        self.mu_head = nn.Linear(d_model, d_model)
        self.logvar_head = nn.Linear(d_model, d_model)
        # 预测头：将 (L, d_model) 投影到 (pred_len, C)
        self.proj_time = nn.Linear(seq_len, pred_len)
        self.proj_feat = nn.Linear(d_model, n_channels)
        # RevIN 风格的输入归一化（按 mask 统计）
        self.eps = 1e-5

    def _revin(self, x: torch.Tensor, mask: torch.Tensor):
        m_sum = mask.sum(dim=1).clamp(min=1.0)
        mean = (x * mask).sum(dim=1, keepdim=True) / m_sum.unsqueeze(1)
        x_c = x - mean
        var = ((x_c * mask) ** 2).sum(dim=1, keepdim=True) / m_sum.unsqueeze(1) + self.eps
        std = torch.sqrt(var)
        x_n = x_c / std
        return x_n, mean, std

    def forward(self, x: torch.Tensor, x_mark=None, mask=None):
        if mask is None:
            mask = torch.ones_like(x)
        x_n, mean, std = self._revin(x, mask)
        h = self.embed(x_n, mask)  # (B, L, d_model)
        h = self.pos(h)
        for layer in self.encoder:
            h = layer(h)
        h = self.norm(h)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h).clamp(-8.0, 8.0)
        if self.training and self.sample_train:
            std_z = torch.exp(0.5 * logvar)
            z = mu + std_z * torch.randn_like(std_z)
        else:
            z = mu
        # KL 对 N(0, I)
        kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean()
        # 预测
        z_t = z.transpose(1, 2)  # (B, D, L)
        z_t = self.proj_time(z_t)  # (B, D, pred_len)
        out = self.proj_feat(z_t.transpose(1, 2))  # (B, pred_len, C)
        out = out * std + mean
        # 返回元组，让训练循环能加 KL 项；为了保持统一接口，附加属性
        self._last_kl = kl
        return out

    def auxiliary_loss(self):
        return self.kl_weight * self._last_kl if hasattr(self, "_last_kl") else 0.0
