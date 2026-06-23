"""CoIFNet（紧凑版）：联合补值-预测网络，共享 backbone 同时输出补值与预测。

参考: K. Tang et al., arXiv:2506.13064, 2025.
原始代码: external/CoIFNet/model/CoIFNet.py

核心思想：
- 输入扩展到 seq_len+pred_len 的"通道"维度（变量+时间协变量+mask），用一个 backbone 同时
  输出整段（历史填补 + 未来预测）。
- 损失为两部分： (a) 历史段在观测位置的重建 MAE  +  (b) 未来段相对真值的 MAE。

简化点：原文 backbone 含多种 mix 模块，这里使用通道-混合 + 时间-混合的轻量 TSMixer 风格。
"""
from __future__ import annotations
import torch
import torch.nn as nn


class _MixerBlock(nn.Module):
    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.ff1 = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden * 2, hidden)
        )
        self.norm2 = nn.LayerNorm(hidden)
        self.ff2 = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden * 2, hidden)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ff1(self.norm1(x))
        x = x + self.ff2(self.norm2(x))
        return x


class _RevIN(nn.Module):
    def __init__(self, n_channels: int, eps: float = 1e-5, mask_aware: bool = True):
        super().__init__()
        self.eps = eps
        self.mask_aware = mask_aware

    def norm(self, x: torch.Tensor, mask: torch.Tensor = None):
        if self.mask_aware and mask is not None:
            m_sum = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            mean = (x * mask).sum(dim=1, keepdim=True) / m_sum
            var = ((x - mean) * mask).pow(2).sum(dim=1, keepdim=True) / m_sum + self.eps
        else:
            mean = x.mean(dim=1, keepdim=True)
            var = x.var(dim=1, keepdim=True, unbiased=False) + self.eps
        std = torch.sqrt(var)
        self.mean = mean
        self.std = std
        return (x - mean) / std

    def denorm(self, x: torch.Tensor):
        return x * self.std + self.mean


class CoIFNet(nn.Module):
    """联合补值预测网络。

    输入：x (B, L, C), mask (B, L, C)
    输出 dict：
        forecast: (B, H, C)
        impute  : (B, L, C)  历史段（替换缺失位置）
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
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.total_len = seq_len + pred_len
        self.n_channels = n_channels
        self.hidden = hidden
        self.impute_weight = impute_weight

        # 输入：concat(x*mask, mask) 沿通道，再加常数 0 占位预测段
        in_dim = n_channels * 2
        self.embed = nn.Linear(in_dim, hidden)
        self.pos = nn.Parameter(torch.randn(1, self.total_len, hidden) * 0.02)
        self.blocks = nn.ModuleList([_MixerBlock(hidden, dropout) for _ in range(n_layers)])
        # 时间维 mix
        self.time_mix = nn.Linear(self.total_len, self.total_len)
        self.norm_out = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, n_channels)
        self.use_revin = use_revin
        if use_revin:
            self.revin = _RevIN(n_channels)

    def forward(self, x: torch.Tensor, x_mark=None, mask=None):
        if mask is None:
            mask = torch.ones_like(x)
        B = x.size(0)
        if self.use_revin:
            x_n = self.revin.norm(x, mask)
        else:
            x_n = x
        # 拼接 [历史, 0-padding] 与 mask
        pad_x = torch.zeros(B, self.pred_len, self.n_channels, device=x.device, dtype=x.dtype)
        pad_m = torch.zeros(B, self.pred_len, self.n_channels, device=x.device, dtype=x.dtype)
        x_full = torch.cat([x_n * mask, pad_x], dim=1)
        m_full = torch.cat([mask, pad_m], dim=1)
        inp = torch.cat([x_full, m_full], dim=-1)  # (B, total, 2C)
        h = self.embed(inp) + self.pos
        for blk in self.blocks:
            h = blk(h)
        # 时间维 mix
        h = h.transpose(1, 2)  # (B, hidden, total)
        h = self.time_mix(h).transpose(1, 2)
        h = self.norm_out(h)
        out_full = self.head(h)  # (B, total, C)
        if self.use_revin:
            out_full = self.revin.denorm(out_full)
        impute = out_full[:, : self.seq_len, :]
        # 仅替换缺失位置
        impute = mask * x + (1.0 - mask) * impute
        forecast = out_full[:, self.seq_len :, :]
        return {"forecast": forecast, "impute": impute}

    def compute_loss(
        self,
        out: dict,
        x_true: torch.Tensor,
        mask: torch.Tensor,
        y_true: torch.Tensor,
        criterion=None,
    ):
        """同时优化历史观测位置的重建 MAE/MSE 与未来预测的 MAE/MSE。"""
        if criterion is None:
            criterion = nn.L1Loss(reduction="none")
        impute = out["impute"]
        forecast = out["forecast"]
        # impute loss on observed positions
        rec = criterion(impute, x_true) * mask
        rec_loss = rec.sum() / mask.sum().clamp(min=1.0)
        # forecast loss on all future positions
        fc_loss = criterion(forecast, y_true).mean()
        return fc_loss + self.impute_weight * rec_loss, rec_loss.detach(), fc_loss.detach()
