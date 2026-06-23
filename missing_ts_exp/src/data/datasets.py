"""统一的多变量长序列数据集封装。

约定：
- 所有数据集都按时间顺序切分；不打乱。
- 训练集均值/标准差用于全集标准化，避免未来信息泄露。
- 数据集对象在每个 epoch 开始时按 (seq_idx, missing_seed) 注入缺失，
  返回 (x_raw, x_obs, mask, x_mark, y, y_mark)。
  x_raw  : 真实历史值（已标准化），用于评估补值误差
  x_obs  : 缺失处理后的输入（mask=0 处为 NaN，由模型/Wrapper 自行填补）
  mask   : (L, C) 0/1，1=有观测
  x_mark : 时间特征
  y      : 未来真值（完整）
  y_mark : 预测段时间特征

我们仅对输入窗口注入缺失，预测目标保持完整。
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .timefeatures import time_features
from .missing import inject_missing
from ..utils.constants import (
    DATASETS,
    ETT_HOURLY_SPLIT,
    ETT_MINUTELY_SPLIT,
)


_RAW_CACHE: dict = {}


def _load_raw(name: str):
    """读 CSV，返回 (values: (T, C) float32, dates: pd.DatetimeIndex, n_features)。"""
    if name in _RAW_CACHE:
        return _RAW_CACHE[name]
    meta = DATASETS[name]
    df = pd.read_csv(meta["path"])
    date_col = meta["date_col"]
    dates = pd.to_datetime(df[date_col].values)
    # 全部其他列均视为变量
    feat_cols = [c for c in df.columns if c != date_col]
    values = df[feat_cols].values.astype(np.float32)
    _RAW_CACHE[name] = (values, dates, len(feat_cols))
    return _RAW_CACHE[name]


def _split_indices(name: str, T: int):
    """返回 (train_end, val_end, test_end)，下标都在 [0, T] 闭半开区间表达。"""
    meta = DATASETS[name]
    if meta["split"] == "ett_hourly":
        s = ETT_HOURLY_SPLIT
        return s["train_end"], s["val_end"], s["test_end"]
    if meta["split"] == "ett_minutely":
        s = ETT_MINUTELY_SPLIT
        return s["train_end"], s["val_end"], s["test_end"]
    if meta["split"] == "ratio":
        r_train, r_val, _ = meta["ratio"]
        train_end = int(T * r_train)
        val_end = train_end + int(T * r_val)
        return train_end, val_end, T
    raise ValueError(meta["split"])


def get_standardizer(name: str):
    """返回训练集统计量 (mean, std)，形状 (C,)，并缓存。"""
    values, _, _ = _load_raw(name)
    train_end, _, _ = _split_indices(name, len(values))
    arr = values[:train_end]
    mean = arr.mean(0)
    std = arr.std(0) + 1e-8
    return mean.astype(np.float32), std.astype(np.float32)


class MissingForecastDataset(Dataset):
    """滑动窗口数据集。

    参数
    ----
    name            : 数据集名（见 DATASETS）
    flag            : 'train' / 'val' / 'test'
    seq_len, pred_len : 输入/预测长度
    missing_type    : 缺失类型，'none' 表示完整输入
    missing_rate    : 缺失率（0-1）
    base_seed       : 用于派生缺失生成 RNG，使得跨 epoch 同一 sample 缺失模式不同
                       但跨实验相同 seed 可复现
    seg_lengths     : 连续片段缺失候选长度
    """

    def __init__(
        self,
        name: str,
        flag: str = "train",
        seq_len: int = 96,
        pred_len: int = 96,
        missing_type: str = "none",
        missing_rate: float = 0.0,
        base_seed: int = 2024,
        seg_lengths=(12, 24, 48),
        deterministic_missing: bool = True,
    ):
        assert flag in ("train", "val", "test")
        self.name = name
        self.flag = flag
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.missing_type = missing_type
        self.missing_rate = float(missing_rate)
        self.base_seed = int(base_seed)
        self.seg_lengths = tuple(seg_lengths)
        self.deterministic_missing = deterministic_missing

        values, dates, n_features = _load_raw(name)
        self.n_features = n_features

        train_end, val_end, test_end = _split_indices(name, len(values))
        meta = DATASETS[name]
        self.freq = meta["freq"]

        # 标准化（用训练集统计量）
        mean, std = get_standardizer(name)
        values = (values - mean) / std
        self.mean = mean
        self.std = std

        if flag == "train":
            border1, border2 = 0, train_end
        elif flag == "val":
            # 验证集需要前面 seq_len 的滑动窗口
            border1 = train_end - self.seq_len
            border2 = val_end
        else:
            border1 = val_end - self.seq_len
            border2 = test_end

        self.data = values[border1:border2]
        self.dates = dates[border1:border2]
        self.time_marks = time_features(self.dates, freq=self.freq)
        # 起始索引：i 处取 [i, i+seq_len) 输入和 [i+seq_len, i+seq_len+pred_len) 标签
        self.num_samples = len(self.data) - self.seq_len - self.pred_len + 1
        if self.num_samples <= 0:
            raise ValueError(
                f"dataset {name} flag={flag} too short: {len(self.data)} for seq_len={seq_len} pred_len={pred_len}"
            )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        s_begin = idx
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        x = self.data[s_begin:s_end]  # (L, C)
        y = self.data[r_begin:r_end]  # (H, C)
        x_mark = self.time_marks[s_begin:s_end]
        y_mark = self.time_marks[r_begin:r_end]

        # 缺失注入：对每个样本独立采样掩码
        # 训练阶段每个 epoch 应该有不同的掩码 -> 通过 (base_seed, epoch, idx) 控制
        # 此处仅按 (base_seed, idx) 派生，避免数据集需要持有 epoch 状态；
        # 训练时通过 set_epoch_seed 动态切换 base_seed_offset 即可。
        if self.missing_type == "none" or self.missing_rate <= 0.0:
            mask = np.ones_like(x, dtype=np.float32)
        else:
            sub_seed = (self.base_seed * 1315423911 + idx) & 0xFFFFFFFF
            mask = inject_missing(
                shape=x.shape,
                missing_type=self.missing_type,
                missing_rate=self.missing_rate,
                seg_lengths=self.seg_lengths,
                seed=int(sub_seed),
            ).astype(np.float32)

        x_obs = x * mask  # 0 处填 0；下游模块如果需要 NaN 用 mask 区分
        return {
            "x_raw": torch.from_numpy(x).float(),
            "x_obs": torch.from_numpy(x_obs).float(),
            "mask": torch.from_numpy(mask).float(),
            "x_mark": torch.from_numpy(x_mark).float(),
            "y": torch.from_numpy(y).float(),
            "y_mark": torch.from_numpy(y_mark).float(),
        }

    def set_epoch_seed_offset(self, offset: int):
        """训练阶段每个 epoch 改变 base_seed 偏移，让缺失模式逐 epoch 变化。"""
        self.base_seed = int(self.base_seed) ^ (int(offset) * 2654435761 & 0xFFFFFFFF)


def collate(batch):
    """默认 collate，把字典里的张量按第 0 维拼起来。"""
    out = {}
    for k in batch[0]:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out
