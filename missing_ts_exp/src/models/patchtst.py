"""PatchTST 紧凑实现（通道独立 + Patch 化 + Transformer encoder）。

输入: x (B, L, C)，沿通道维度独立编码；每个通道切成 patches。
输出: y (B, H, C)
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class _RevIN(nn.Module):
    def __init__(self, n_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(n_channels))
            self.beta = nn.Parameter(torch.zeros(n_channels))

    def forward(self, x: torch.Tensor, mode: str):
        # x: (B, L, C)
        if mode == "norm":
            self.mean = x.mean(dim=1, keepdim=True).detach()
            self.std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            x = (x - self.mean) / self.std
            if self.affine:
                x = x * self.gamma + self.beta
            return x
        if mode == "denorm":
            if self.affine:
                x = (x - self.beta) / (self.gamma + 1e-8)
            x = x * self.std + self.mean
            return x
        raise ValueError(mode)


class _AttnBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        a, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout(a))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class PatchTST(nn.Module):
    name = "PatchTST"

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_channels: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 8,
        e_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        use_revin: bool = True,
        time_feat_dim: int = 0,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_channels = n_channels
        self.patch_len = patch_len
        self.stride = stride
        self.use_revin = use_revin
        if use_revin:
            self.revin = _RevIN(n_channels)
        # 计算 patch 数：与原论文一致，先在序列起始处复制 stride 个值再切
        self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
        self.patch_num = (seq_len - patch_len) // stride + 2
        self.embed = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.randn(1, self.patch_num, d_model) * 0.02)
        self.blocks = nn.ModuleList(
            [_AttnBlock(d_model, n_heads, d_ff, dropout) for _ in range(e_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(self.patch_num * d_model, pred_len)

    def _patchify(self, x):
        # x: (B*C, L) -> (B*C, n_patch, patch_len)
        x = self.padding_patch_layer(x.unsqueeze(1)).squeeze(1)
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        return patches

    def forward(self, x: torch.Tensor, x_mark=None, mask=None) -> torch.Tensor:
        B, L, C = x.shape
        if self.use_revin:
            x = self.revin(x, "norm")
        x_t = x.permute(0, 2, 1).contiguous().view(B * C, L)  # (B*C, L)
        patches = self._patchify(x_t)  # (B*C, n_patch, patch_len)
        h = self.embed(patches) + self.pos
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        h_flat = h.reshape(B * C, -1)
        y = self.head(h_flat).view(B, C, self.pred_len).permute(0, 2, 1)  # (B, H, C)
        if self.use_revin:
            y = self.revin(y, "denorm")
        return y
