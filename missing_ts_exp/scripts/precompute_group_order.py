"""离线预计算 MissTSM 方案 A（相关性预分组）的通道重排顺序。

对 Weather / Electricity / Traffic 各跑一次层次聚类，缓存到
results_cache/group_order/{dataset}.json，供 MissTSMPipeline 加载。
"""
from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.grouping import get_or_compute_channel_order

DATASETS = ["Weather", "Electricity", "Traffic"]


def main():
    for ds in DATASETS:
        order = get_or_compute_channel_order(ds, force=True)
        print(f"{ds}: C={len(order)}")
        print(f"  order[:20] = {order[:20]}")


if __name__ == "__main__":
    main()
