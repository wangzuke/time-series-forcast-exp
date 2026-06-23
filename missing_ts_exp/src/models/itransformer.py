"""iTransformer 倒置 Transformer 紧凑实现。

输入: x_enc (B, L, C), x_mark (B, L, T_feat) 可选
输出: y (B, H, C)
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn


class _DataEmbeddingInverted(nn.Module):
    def __init__(self, seq_len: int, d_model: int, time_feat_dim: int = 0, dropout: float = 0.1):
        super().__init__()
        in_dim = seq_len  # 每个变量是一个 token，其特征是它的历史长度
        self.value_embedding = nn.Linear(in_dim, d_model)
        self.time_feat_dim = time_feat_dim
        if time_feat_dim > 0:
            # 时间协变量也作为额外 token（覆盖 seq_len 的时序），先做 Linear
            self.time_embedding = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, x_mark=None) -> torch.Tensor:
        # x: (B, L, C) -> (B, C, L)
        x_inv = x.permute(0, 2, 1)
        var_tokens = self.value_embedding(x_inv)  # (B, C, d_model)
        if self.time_feat_dim > 0 and x_mark is not None:
            # x_mark: (B, L, T_feat) -> (B, T_feat, L)
            tm = x_mark.permute(0, 2, 1)
            time_tokens = self.time_embedding(tm)  # (B, T_feat, d_model)
            var_tokens = torch.cat([var_tokens, time_tokens], dim=1)
        return self.dropout(var_tokens)


class _SelfAttn(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        H = self.n_heads
        q = self.q(x).view(B, N, H, self.d_k).transpose(1, 2)
        k = self.k(x).view(B, N, H, self.d_k).transpose(1, 2)
        v = self.v(x).view(B, N, H, self.d_k).transpose(1, 2)
        scale = 1.0 / math.sqrt(self.d_k)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # (B, H, N, d_k)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out(out)


class _EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = _SelfAttn(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.attn(x)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class iTransformer(nn.Module):
    name = "iTransformer"

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
        time_feat_dim: int = 0,
        use_norm: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_channels = n_channels
        self.use_norm = use_norm
        self.embed = _DataEmbeddingInverted(seq_len, d_model, time_feat_dim, dropout)
        self.encoder = nn.ModuleList(
            [_EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(e_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.projector = nn.Linear(d_model, pred_len)

    def forward(self, x: torch.Tensor, x_mark=None, mask=None) -> torch.Tensor:
        if self.use_norm:
            means = x.mean(1, keepdim=True).detach()
            x_n = x - means
            stdev = torch.sqrt(x_n.var(1, keepdim=True, unbiased=False) + 1e-5)
            x_n = x_n / stdev
        else:
            x_n = x
        h = self.embed(x_n, x_mark)
        for layer in self.encoder:
            h = layer(h)
        h = self.norm(h)
        # 投影到 pred_len，仅保留前 N 个变量 token（丢弃时间协变量 token）
        out = self.projector(h)  # (B, N_token, pred_len)
        out = out[:, : self.n_channels, :].permute(0, 2, 1)  # (B, pred_len, C)
        if self.use_norm:
            out = out * stdev[:, 0, :].unsqueeze(1) + means[:, 0, :].unsqueeze(1)
        return out
