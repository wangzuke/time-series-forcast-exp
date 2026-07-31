"""相关性驱动的通道分组（方案 A：见 docs/实验计划0709.md §2；方案 D：见 docs/实验计划0710.md §2）。

核心思路：MissTSM 的 grouped_q 变体把 C 个通道按固定序号连续切片分组。
这里改为先算一次通道相关矩阵，做层次聚类得到一个"重排顺序"，
使得相关性强的通道在这个顺序里彼此靠近，再复用原有的连续切片逻辑——
不需要改动分组循环本身，只需要在切片前对通道维做一次固定的 index 重排。

方案 A（compute_channel_order）用完整、无缺失的训练序列计算相关性。
方案 D（compute_channel_order_observed）额外注入一次固定种子的缺失掩码，
用"观测到的"（缺失位置填 0）训练序列计算相关性，用于检验 0709 报告里
Weather continuous_segment 退化是否源于静态先验与实际观测状态脱节。

只使用训练集统计量（与项目标准化约定一致），不使用验证/测试集。
"""
from __future__ import annotations
import os
import json
import numpy as np

from .datasets import _load_raw, _split_indices
from .missing import inject_missing

_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "results_cache", "group_order",
)
_CACHE_DIR_OBS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "results_cache", "group_order_obs",
)


def _cluster_order_from_array(arr: np.ndarray, n_features: int) -> list[int]:
    """给定 (T, C) 数组，返回层次聚类得到的通道重排顺序。"""
    if n_features <= 2:
        return list(range(n_features))

    corr = np.corrcoef(arr, rowvar=False)  # (C, C)
    corr = np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0)
    dist = 1.0 - np.abs(corr)
    np.fill_diagonal(dist, 0.0)
    dist = np.clip((dist + dist.T) / 2.0, 0.0, None)  # 对称化，修正浮点误差
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import squareform

        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method="average", optimal_ordering=True)
        order = leaves_list(Z)
        return [int(i) for i in order]
    except ImportError:
        return _greedy_corr_order(corr)


def _greedy_corr_order(corr: np.ndarray) -> list[int]:
    """无 scipy 环境下的确定性 fallback。

    从平均绝对相关性最高的通道开始，每次接上与当前末尾最相关的未访问通道。
    它不是层次聚类的完全替代，但能稳定地把相关通道排近，保证 grouped_q 变体可运行。
    """
    C = corr.shape[0]
    sim = np.abs(corr)
    np.fill_diagonal(sim, 0.0)
    start = int(np.argmax(sim.mean(axis=1)))
    order = [start]
    unused = set(range(C))
    unused.remove(start)
    while unused:
        last = order[-1]
        nxt = max(unused, key=lambda j: (float(sim[last, j]), -int(j)))
        order.append(int(nxt))
        unused.remove(nxt)
    return order


def compute_channel_order(dataset_name: str) -> list[int]:
    """返回长度为 C 的通道重排顺序（0..C-1 的一个排列）。

    做法：训练集原始序列 -> 皮尔逊相关矩阵 -> 距离 = 1-|corr| -> 层次聚类
    （average linkage + optimal leaf ordering）-> 叶子顺序即为重排顺序。
    """
    values, _, n_features = _load_raw(dataset_name)
    train_end, _, _ = _split_indices(dataset_name, len(values))
    arr = values[:train_end]  # (T_train, C)
    return _cluster_order_from_array(arr, n_features)


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


def compute_channel_order_observed(
    dataset_name: str, missing_type: str, missing_rate: float, mask_seed: int = 0,
) -> list[int]:
    """方案 D：用注入一次固定种子缺失后的观测数据（缺失位置填 0）计算相关性重排。

    与 compute_channel_order 唯一的区别是：先对完整训练序列注入一次
    (missing_type, missing_rate) 缺失（固定 mask_seed，不随训练 seed 变化），
    缺失位置填 0 后再算相关矩阵，其余聚类流程完全一致。
    """
    values, _, n_features = _load_raw(dataset_name)
    train_end, _, _ = _split_indices(dataset_name, len(values))
    arr = values[:train_end]  # (T_train, C)

    if missing_rate <= 0 or missing_type in ("none", None):
        return _cluster_order_from_array(arr, n_features)

    mask = inject_missing(
        shape=arr.shape,
        missing_type=missing_type,
        missing_rate=missing_rate,
        seed=mask_seed,
    )
    arr_obs = arr * mask
    return _cluster_order_from_array(arr_obs, n_features)


def get_or_compute_channel_order_observed(
    dataset_name: str, missing_type: str, missing_rate: float,
    mask_seed: int = 0, force: bool = False,
) -> list[int]:
    """带缓存版本：results_cache/group_order_obs/{dataset}__{missing_type}_{rate_int}.json"""
    os.makedirs(_CACHE_DIR_OBS, exist_ok=True)
    rate_int = int(round(missing_rate * 100))
    cache_path = os.path.join(_CACHE_DIR_OBS, f"{dataset_name}__{missing_type}_{rate_int}.json")
    if not force and os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)["order"]
    order = compute_channel_order_observed(dataset_name, missing_type, missing_rate, mask_seed)
    with open(cache_path, "w") as f:
        json.dump({
            "dataset": dataset_name, "missing_type": missing_type,
            "missing_rate": missing_rate, "mask_seed": mask_seed, "order": order,
        }, f)
    return order
