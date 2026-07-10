"""相关性驱动的通道分组（方案 A：见 docs/实验计划0709.md §2）。

核心思路：MissTSM 的 grouped_q 变体把 C 个通道按固定序号连续切片分组。
这里改为先用训练集统计量算一次通道相关矩阵，做层次聚类得到一个"重排顺序"，
使得相关性强的通道在这个顺序里彼此靠近，再复用原有的连续切片逻辑——
不需要改动分组循环本身，只需要在切片前对通道维做一次固定的 index 重排。

只使用训练集统计量（与项目标准化约定一致），不使用验证/测试集。
"""
from __future__ import annotations
import os
import json
import numpy as np

from .datasets import _load_raw, _split_indices

_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "results_cache", "group_order",
)


def compute_channel_order(dataset_name: str) -> list[int]:
    """返回长度为 C 的通道重排顺序（0..C-1 的一个排列）。

    做法：训练集原始序列 -> 皮尔逊相关矩阵 -> 距离 = 1-|corr| -> 层次聚类
    （average linkage + optimal leaf ordering）-> 叶子顺序即为重排顺序。
    """
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform

    values, _, n_features = _load_raw(dataset_name)
    train_end, _, _ = _split_indices(dataset_name, len(values))
    arr = values[:train_end]  # (T_train, C)

    if n_features <= 2:
        return list(range(n_features))

    corr = np.corrcoef(arr, rowvar=False)  # (C, C)
    corr = np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0)
    dist = 1.0 - np.abs(corr)
    np.fill_diagonal(dist, 0.0)
    dist = np.clip((dist + dist.T) / 2.0, 0.0, None)  # 对称化，修正浮点误差
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average", optimal_ordering=True)
    order = leaves_list(Z)
    return [int(i) for i in order]


def get_or_compute_channel_order(dataset_name: str, force: bool = False) -> list[int]:
    """带缓存版本：results_cache/group_order/{dataset}.json"""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(_CACHE_DIR, f"{dataset_name}.json")
    if not force and os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)["order"]
    order = compute_channel_order(dataset_name)
    with open(cache_path, "w") as f:
        json.dump({"dataset": dataset_name, "order": order}, f)
    return order
