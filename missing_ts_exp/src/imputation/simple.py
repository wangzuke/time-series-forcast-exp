"""简单填补策略（按变量逐列）。

输入：
  x      : (B, L, C) 带 NaN 或 0 的序列（使用 mask 区分缺失）
  mask   : (B, L, C) 1=有观测，0=缺失
  global_mean : (C,) 训练集均值（用于均值填补；标准化后均值≈0）

输出：filled (B, L, C)
"""
from __future__ import annotations
import torch


def fill_zero(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """直接置 0（在标准化空间下相当于均值填补，因为训练集均值=0）。"""
    return x * mask


def fill_global_mean(x: torch.Tensor, mask: torch.Tensor, global_mean=None) -> torch.Tensor:
    """用全局均值替代缺失位置。

    标准化后训练集均值 ≈ 0，所以 global_mean=None 时退化为 fill_zero。
    """
    if global_mean is None:
        return x * mask
    if isinstance(global_mean, torch.Tensor):
        gm = global_mean.to(x.device, x.dtype)
    else:
        gm = torch.as_tensor(global_mean, device=x.device, dtype=x.dtype)
    # gm shape (C,)
    gm = gm.view(1, 1, -1)
    return x * mask + gm * (1.0 - mask)


def fill_forward(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """沿时间维度做前向填补；窗口起始缺失用 0 兜底。"""
    B, L, C = x.shape
    out = x.clone()
    last = torch.zeros(B, C, device=x.device, dtype=x.dtype)
    has = torch.zeros(B, C, device=x.device, dtype=torch.bool)
    for t in range(L):
        m_t = mask[:, t, :].bool()
        cur = x[:, t, :]
        last = torch.where(m_t, cur, last)
        has = has | m_t
        # 缺失位置：若曾出现过观测就用 last，否则保留 0
        fill_val = torch.where(has, last, torch.zeros_like(last))
        out[:, t, :] = torch.where(m_t, cur, fill_val)
    return out


def fill_linear_interp(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """沿时间维度做线性插值，两端缺失退化为最近邻/0。"""
    B, L, C = x.shape
    out = x.clone()
    # 把缺失置为 nan 便于追踪（仅用张量结构）
    # 先做前向填，再后向填，再线性插值。简单实现：对每个 (b, c) 独立处理。
    obs_mask = mask.bool()
    # forward fill
    fwd = fill_forward(x, mask)
    # backward fill：反向时间再前向填一次
    bwd = fill_forward(torch.flip(x, dims=[1]), torch.flip(mask, dims=[1]))
    bwd = torch.flip(bwd, dims=[1])
    # 对缺失位置取 fwd 与 bwd 的位置加权平均（按到上一/下一观测的距离）
    # 距离计算
    t_index = torch.arange(L, device=x.device).view(1, L, 1).float()
    # 距离上一个观测
    big = float(L * 10)
    fwd_t = torch.where(obs_mask, t_index.expand(B, L, C), torch.full_like(t_index.expand(B, L, C), -big))
    fwd_t = fwd_t.cummax(dim=1).values  # 上一个观测的时间
    bwd_t = torch.where(
        obs_mask, t_index.expand(B, L, C), torch.full_like(t_index.expand(B, L, C), big)
    )
    bwd_t = torch.flip(torch.flip(bwd_t, dims=[1]).cummin(dim=1).values, dims=[1])

    # 权重
    denom = (bwd_t - fwd_t).clamp(min=1e-6)
    w_fwd = (bwd_t - t_index) / denom
    w_bwd = (t_index - fwd_t) / denom
    interp = w_fwd * fwd + w_bwd * bwd
    # 当两侧都没有观测时（极端情况），保持 0
    valid = (fwd_t > -big * 0.5) & (bwd_t < big * 0.5)
    interp = torch.where(valid, interp, torch.zeros_like(interp))
    # 仅替换缺失位置
    out = torch.where(obs_mask, x, interp)
    return out


SIMPLE_IMPUTERS = {
    "mean": fill_global_mean,
    "zero": fill_zero,
    "forward": fill_forward,
    "linear": fill_linear_interp,
}


def apply_simple_imputer(name: str, x: torch.Tensor, mask: torch.Tensor, global_mean=None) -> torch.Tensor:
    if name not in SIMPLE_IMPUTERS:
        raise ValueError(f"unknown imputer {name}")
    if name == "mean":
        return fill_global_mean(x, mask, global_mean)
    return SIMPLE_IMPUTERS[name](x, mask)
