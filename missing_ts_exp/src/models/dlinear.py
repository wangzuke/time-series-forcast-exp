"""DLinear：分解-线性预测器。

输入:  x (B, L, C)
输出:  y (B, H, C)
"""
from __future__ import annotations
import torch
import torch.nn as nn


class _MovingAvg(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        pad = (self.kernel_size - 1) // 2
        front = x[:, :1, :].repeat(1, pad, 1)
        end = x[:, -1:, :].repeat(1, pad, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class _SeriesDecomp(nn.Module):
    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.ma = _MovingAvg(kernel_size)

    def forward(self, x: torch.Tensor):
        trend = self.ma(x)
        return x - trend, trend


class DLinear(nn.Module):
    name = "DLinear"

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_channels: int,
        kernel_size: int = 25,
        individual: bool = False,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_channels = n_channels
        self.individual = individual
        self.decomp = _SeriesDecomp(kernel_size)
        if individual:
            self.linear_s = nn.ModuleList(
                [nn.Linear(seq_len, pred_len) for _ in range(n_channels)]
            )
            self.linear_t = nn.ModuleList(
                [nn.Linear(seq_len, pred_len) for _ in range(n_channels)]
            )
        else:
            self.linear_s = nn.Linear(seq_len, pred_len)
            self.linear_t = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor, x_mark=None, mask=None) -> torch.Tensor:
        # x: (B, L, C)
        s, t = self.decomp(x)
        s = s.transpose(1, 2)  # (B, C, L)
        t = t.transpose(1, 2)
        if self.individual:
            outs = []
            outt = []
            for i in range(self.n_channels):
                outs.append(self.linear_s[i](s[:, i]))
                outt.append(self.linear_t[i](t[:, i]))
            s_out = torch.stack(outs, dim=1)
            t_out = torch.stack(outt, dim=1)
        else:
            s_out = self.linear_s(s)
            t_out = self.linear_t(t)
        y = (s_out + t_out).transpose(1, 2)  # (B, H, C)
        return y
