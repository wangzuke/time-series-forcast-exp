"""离线预计算 MissTSM 方案 D（观测数据相关性分组）的通道重排顺序。

对 Weather / Electricity / Traffic × {random_point, continuous_segment} ×
{0.3, 0.5, 0.7} 共 18 个组合各跑一次，缓存到
results_cache/group_order_obs/{dataset}__{missing_type}_{rate_int}.json，
供 MissTSMPipeline 加载。详见 docs/实验计划0710.md §2。
"""
from __future__ import annotations
import sys
import os
import itertools

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.grouping import get_or_compute_channel_order_observed

DATASETS = ["Weather", "Electricity", "Traffic"]
MISS_TYPES = ["random_point", "continuous_segment"]
MISS_RATES = [0.3, 0.5, 0.7]


def main():
    for ds, mt, mr in itertools.product(DATASETS, MISS_TYPES, MISS_RATES):
        order = get_or_compute_channel_order_observed(ds, mt, mr, force=True)
        print(f"{ds} {mt} {mr}: C={len(order)}")
        print(f"  order[:20] = {order[:20]}")


if __name__ == "__main__":
    main()
