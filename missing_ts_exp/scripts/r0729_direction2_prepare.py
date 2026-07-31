"""方向2实验准备：检查数据并预计算通道分组缓存。

该脚本不训练模型，只做两件事：
1. 确认实验计划所需数据集 CSV 可读，并输出基本规模。
2. 为 PMA / grouped MissTSM 变体预计算完整相关分组与观测相关分组缓存。

用法：
    cd missing_ts_exp
    python scripts/r0729_direction2_prepare.py --datasets ETTh1 Weather Electricity Traffic
"""
from __future__ import annotations

import argparse
import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from missing_ts_exp.src.data.datasets import _load_raw, _split_indices  # noqa: E402
from missing_ts_exp.src.data.grouping import (  # noqa: E402
    get_or_compute_channel_order,
    get_or_compute_channel_order_observed,
)
from missing_ts_exp.src.utils.constants import DATASETS  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ETTh1", "Weather", "Electricity", "Traffic"])
    ap.add_argument(
        "--missing_types",
        nargs="+",
        default=["random_point", "continuous_segment", "variable_channel", "mixed"],
    )
    ap.add_argument("--missing_rates", nargs="+", type=float, default=[0.1, 0.3, 0.5, 0.7])
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    print("[direction2] data check")
    for ds in args.datasets:
        if ds not in DATASETS:
            raise KeyError(f"unknown dataset: {ds}")
        path = DATASETS[ds]["path"]
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        values, dates, n_features = _load_raw(ds)
        train_end, val_end, test_end = _split_indices(ds, len(values))
        print(
            f"  {ds}: rows={len(values)} channels={n_features} "
            f"train_end={train_end} val_end={val_end} test_end={test_end} path={path}"
        )
        order = get_or_compute_channel_order(ds, force=args.force)
        print(f"    full-corr group order cached: len={len(order)}")
        for mt in args.missing_types:
            for r in args.missing_rates:
                obs_order = get_or_compute_channel_order_observed(
                    ds, mt, r, force=args.force,
                )
                print(f"    obs-corr order cached: {mt}:{r:.1f} len={len(obs_order)}")
    print("[direction2] preparation done")


if __name__ == "__main__":
    main()
