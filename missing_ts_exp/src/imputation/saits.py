"""SAITS：Self-Attention-based Imputation for Time Series（紧凑独立版本）。

参考：W. Du et al., Expert Systems with Applications, 2023.
原始代码：external/SAITS/modeling/{saits,layers}.py

接口：
    forward(X, mask) -> dict
        imputed:  (B, L, C) — 用观测替换缺失位置后的完整序列
        Xtilde1/2/3: 三个重建
    fit_step(X_obs, mask, X_true, indicating_mask) — 一步训练损失
"""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mae(pred, target, mask):
    """带掩码的 MAE：仅在 mask=1 处计算。"""
    num = (torch.abs(pred - target) * mask).sum()
    den = mask.sum().clamp(min=1e-8)
    return num / den


class _MHA(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, attn_dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_q = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_k = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_v = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, x, attn_mask=None):
        B, L, D = x.shape
        H = self.n_head
        q = self.w_q(x).view(B, L, H, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(B, L, H, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(B, L, H, self.d_v).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 1, -1e9)
        attn = self.dropout(torch.softmax(attn, dim=-1))
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.fc(out), attn


class _EncoderLayer(nn.Module):
    def __init__(self, d_time, d_model, d_inner, n_head, d_k, d_v, dropout, diag_mask=True):
        super().__init__()
        self.diag_mask = diag_mask
        self.d_time = d_time
        self.norm = nn.LayerNorm(d_model)
        self.attn = _MHA(n_head, d_model, d_k, d_v, dropout)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_inner), nn.ReLU(),
            nn.Linear(d_inner, d_model), nn.Dropout(dropout)
        )

    def forward(self, x):
        if self.diag_mask:
            mask = torch.eye(self.d_time, device=x.device)
        else:
            mask = None
        residual = x
        x_norm = self.norm(x)
        attn_out, attn_w = self.attn(x_norm, mask)
        x = residual + self.dropout(attn_out)
        x = x + self.ffn(x)
        return x, attn_w


class _PosEnc(nn.Module):
    def __init__(self, d_model, n_position):
        super().__init__()
        pe = np.zeros((n_position, d_model), dtype=np.float32)
        pos = np.arange(n_position).reshape(-1, 1)
        div = np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(pos * div)
        pe[:, 1::2] = np.cos(pos * div)
        self.register_buffer("pe", torch.from_numpy(pe).unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class SAITS(nn.Module):
    """SAITS 补值器（默认两层 DMSA block）。

    n_groups, n_group_inner_layers 与原论文一致。本封装固定使用 'inner_group' 共享策略。
    """

    name = "SAITS"

    def __init__(
        self,
        d_time: int,
        d_feature: int,
        d_model: int = 256,
        d_inner: int = 128,
        n_head: int = 4,
        d_k: int = 64,
        d_v: int = 64,
        n_groups: int = 2,
        n_group_inner_layers: int = 1,
        dropout: float = 0.1,
        input_with_mask: bool = True,
        diag_mask: bool = True,
    ):
        super().__init__()
        self.d_time = d_time
        self.d_feature = d_feature
        self.input_with_mask = input_with_mask
        actual_in = d_feature * 2 if input_with_mask else d_feature

        self.dropout = nn.Dropout(dropout)
        self.pos = _PosEnc(d_model, n_position=d_time)

        self.n_groups = n_groups
        self.n_group_inner = n_group_inner_layers
        self.stack1 = nn.ModuleList(
            [_EncoderLayer(d_time, d_model, d_inner, n_head, d_k, d_v, dropout, diag_mask)
             for _ in range(n_groups)]
        )
        self.stack2 = nn.ModuleList(
            [_EncoderLayer(d_time, d_model, d_inner, n_head, d_k, d_v, dropout, diag_mask)
             for _ in range(n_groups)]
        )
        self.embed1 = nn.Linear(actual_in, d_model)
        self.embed2 = nn.Linear(actual_in, d_model)
        self.reduce_z = nn.Linear(d_model, d_feature)
        self.reduce_beta = nn.Linear(d_model, d_feature)
        self.reduce_gamma = nn.Linear(d_feature, d_feature)
        self.weight_combine = nn.Linear(d_feature + d_time, d_feature)

    def impute(self, X, mask):
        # 1st DMSA
        inp1 = torch.cat([X, mask], dim=-1) if self.input_with_mask else X
        h = self.dropout(self.pos(self.embed1(inp1)))
        for layer in self.stack1:
            for _ in range(self.n_group_inner):
                h, _ = layer(h)
        Xt1 = self.reduce_z(h)
        Xp = mask * X + (1 - mask) * Xt1

        # 2nd DMSA
        inp2 = torch.cat([Xp, mask], dim=-1) if self.input_with_mask else Xp
        h = self.pos(self.embed2(inp2))
        attn_w = None
        for layer in self.stack2:
            for _ in range(self.n_group_inner):
                h, attn_w = layer(h)
        Xt2 = self.reduce_gamma(F.relu(self.reduce_beta(h)))

        # combination：attn_w 形状 (B, n_head, L, L)
        if attn_w is not None:
            aw = attn_w.mean(dim=1)  # (B, L, L)
        else:
            aw = torch.zeros(X.size(0), X.size(1), X.size(1), device=X.device)
        combine = torch.sigmoid(
            self.weight_combine(torch.cat([mask, aw], dim=-1))
        )
        Xt3 = (1 - combine) * Xt2 + combine * Xt1
        imputed = mask * X + (1 - mask) * Xt3
        return imputed, Xt1, Xt2, Xt3

    def forward(self, X, mask):
        imputed, Xt1, Xt2, Xt3 = self.impute(X, mask)
        return {"imputed": imputed, "Xt1": Xt1, "Xt2": Xt2, "Xt3": Xt3}

    def compute_loss(self, X_obs, mask_obs, X_true, indicating_mask, mit_weight: float = 1.0):
        """SAITS 训练损失。

        X_obs : (B, L, C) 部分观测序列（缺失位置已置 0）
        mask_obs : 当前实际可见的观测掩码
        X_true : (B, L, C) 真实值（仅在被人为 MIT 丢弃位置使用监督）
        indicating_mask : (B, L, C) 1 表示该位置是被 MIT 人为屏蔽的位置
        """
        out = self.forward(X_obs, mask_obs)
        Xt1, Xt2, Xt3 = out["Xt1"], out["Xt2"], out["Xt3"]
        # 重建损失（在可见位置）
        rec = (masked_mae(Xt1, X_true, mask_obs)
               + masked_mae(Xt2, X_true, mask_obs)
               + masked_mae(Xt3, X_true, mask_obs)) / 3.0
        # MIT 监督损失（在被人为屏蔽位置）
        if indicating_mask is not None and indicating_mask.sum() > 0:
            imp = masked_mae(Xt3, X_true, indicating_mask)
        else:
            imp = torch.tensor(0.0, device=X_obs.device)
        loss = rec + mit_weight * imp
        return loss, rec.detach(), imp.detach(), out["imputed"]


def random_mit_mask(mask_obs: torch.Tensor, p: float = 0.2, generator=None) -> torch.Tensor:
    """从当前可见位置随机再选 p 比例丢弃，作为 MIT 监督目标位置。

    返回 indicating_mask: (B, L, C) 1 表示该位置由真值变为人为屏蔽。
    返回 new_mask_obs: 训练时实际可见的掩码（原 mask 中 indicating_mask 为 1 的位置再设为 0）。
    """
    rnd = torch.rand_like(mask_obs)
    indicating = ((rnd < p) & (mask_obs > 0.5)).float()
    new_mask = mask_obs * (1 - indicating)
    return indicating, new_mask
