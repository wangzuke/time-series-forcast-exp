"""缺失掩码注入器：四种缺失模式。

返回形状 (L, C) 的 0/1 numpy 数组，1 表示有观测，0 表示缺失。
"""
from __future__ import annotations
import numpy as np


def _random_point_mask(L: int, C: int, rate: float, rng: np.random.Generator) -> np.ndarray:
    mask = (rng.random(size=(L, C)) >= rate).astype(np.float32)
    return mask


def _continuous_segment_mask(
    L: int, C: int, rate: float, rng: np.random.Generator, seg_lengths=(12, 24, 48)
) -> np.ndarray:
    """对随机选择的变量，连续若干步置为缺失，直到达到目标缺失率。"""
    mask = np.ones((L, C), dtype=np.float32)
    target = int(round(rate * L * C))
    missing = 0
    # 上限循环次数，避免极端情况下无限循环
    max_iter = max(1000, target // max(1, min(seg_lengths)) + 100)
    it = 0
    while missing < target and it < max_iter:
        it += 1
        seg = int(rng.choice(seg_lengths))
        seg = min(seg, L)
        c = int(rng.integers(0, C))
        t0 = int(rng.integers(0, L - seg + 1))
        prev = int(mask[t0 : t0 + seg, c].sum())
        if prev == 0:
            continue
        mask[t0 : t0 + seg, c] = 0.0
        missing += prev
    return mask


def _variable_channel_mask(
    L: int, C: int, rate: float, rng: np.random.Generator
) -> np.ndarray:
    """随机选若干变量，整段输入窗口都置为缺失。"""
    mask = np.ones((L, C), dtype=np.float32)
    if C <= 0:
        return mask
    # 选 round(rate*C) 个通道作为完全缺失
    n_miss = int(round(rate * C))
    n_miss = max(1, min(n_miss, C - 1)) if rate > 0 else 0
    if n_miss == 0:
        return mask
    cols = rng.choice(C, size=n_miss, replace=False)
    mask[:, cols] = 0.0
    return mask


def _mixed_mask(
    L: int, C: int, rate: float, rng: np.random.Generator, seg_lengths=(12, 24, 48)
) -> np.ndarray:
    """按 50% / 30% / 20% 比例拼合三种缺失。

    实现方式：分别采样三个掩码，各自的目标缺失率为整体率的 0.5/0.3/0.2；
    取交集（任一为缺失则为缺失），逼近整体缺失率。
    """
    m1 = _random_point_mask(L, C, rate * 0.5, rng)
    m2 = _continuous_segment_mask(L, C, rate * 0.3, rng, seg_lengths)
    m3 = _variable_channel_mask(L, C, rate * 0.2, rng)
    return (m1 * m2 * m3).astype(np.float32)


def inject_missing(
    shape,
    missing_type: str,
    missing_rate: float,
    seg_lengths=(12, 24, 48),
    seed: int = 0,
) -> np.ndarray:
    L, C = shape
    rng = np.random.default_rng(seed)
    if missing_rate <= 0:
        return np.ones((L, C), dtype=np.float32)
    t = missing_type
    if t == "random_point":
        return _random_point_mask(L, C, missing_rate, rng)
    if t == "continuous_segment":
        return _continuous_segment_mask(L, C, missing_rate, rng, seg_lengths)
    if t == "variable_channel":
        return _variable_channel_mask(L, C, missing_rate, rng)
    if t == "mixed":
        return _mixed_mask(L, C, missing_rate, rng, seg_lengths)
    raise ValueError(f"unknown missing_type: {missing_type}")
