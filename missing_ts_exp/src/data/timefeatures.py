"""时间特征：按 iTransformer/PatchTST 标准做法构造，用于位置/时间编码。"""
from __future__ import annotations
import numpy as np
import pandas as pd


def time_features(dates: pd.DatetimeIndex, freq: str = "h") -> np.ndarray:
    """返回 (T, F) 的时间特征矩阵，已归一化到 [-0.5, 0.5]。

    freq 仅用于决定输出的维度数量，与 iTransformer/PatchTST 默认一致：
      - 't' (分钟)  -> 5 维 [minute, hour, weekday, day, month]
      - 'h' (小时)  -> 4 维 [hour, weekday, day, month]
      - 'd' (天)    -> 3 维 [weekday, day, month]
    """
    feats = []
    if freq == "t":
        feats.append((dates.minute / 59.0) - 0.5)
        feats.append((dates.hour / 23.0) - 0.5)
        feats.append((dates.dayofweek / 6.0) - 0.5)
        feats.append((dates.day - 1) / 30.0 - 0.5)
        feats.append((dates.month - 1) / 11.0 - 0.5)
    elif freq == "h":
        feats.append((dates.hour / 23.0) - 0.5)
        feats.append((dates.dayofweek / 6.0) - 0.5)
        feats.append((dates.day - 1) / 30.0 - 0.5)
        feats.append((dates.month - 1) / 11.0 - 0.5)
    elif freq == "d":
        feats.append((dates.dayofweek / 6.0) - 0.5)
        feats.append((dates.day - 1) / 30.0 - 0.5)
        feats.append((dates.month - 1) / 11.0 - 0.5)
    else:
        feats.append((dates.hour / 23.0) - 0.5)
        feats.append((dates.dayofweek / 6.0) - 0.5)
        feats.append((dates.day - 1) / 30.0 - 0.5)
        feats.append((dates.month - 1) / 11.0 - 0.5)
    return np.stack(feats, axis=-1).astype(np.float32)


def time_feature_dim(freq: str) -> int:
    return {"t": 5, "h": 4, "d": 3}.get(freq, 4)
