#!/usr/bin/env python3
"""Build the isolated, per-channel-balanced r=50%, L=H=96 P/B24 suite."""
from __future__ import annotations

import argparse
import pathlib

from protocol import EXP_ROOT, atomic_json_dump


TRAIN_SEEDS = [2024, 2025]
TEST_MASK_SEEDS = [9101, 9102]
BASELINES = ["S4M", "ChannelTokenFormer", "BiTGraph"]
DATASETS = ["ETTh1", "Weather", "Electricity"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=pathlib.Path,
                        default=EXP_ROOT / "configs" / "r50_b24_l96_h96_per_channel_2seeds.json")
    args = parser.parse_args()
    jobs: list[dict] = []
    index = 0
    for baseline in BASELINES:
        for dataset in DATASETS:
            for train_family in ("P", "B"):
                for train_seed in TRAIN_SEEDS:
                    evaluations = [
                        {"test_family": test_family, "block_len": 24, "test_mask_seed": test_seed}
                        for test_family in ("P", "B")
                        for test_seed in TEST_MASK_SEEDS
                    ]
                    jobs.append({
                        "id": f"R50PC_{index:04d}_{baseline}_{dataset}_L96_H96_{train_family}_b24_s{train_seed}",
                        "baseline": baseline,
                        "dataset": dataset,
                        "seq_len": 96,
                        "pred_len": 96,
                        "missing_rate": 0.50,
                        "mask_balance": "per_channel",
                        "train_family": train_family,
                        "train_block_len": 24,
                        "train_seed": train_seed,
                        "evaluations": evaluations,
                    })
                    index += 1
    payload = {
        "protocol": "supplementary_2x2_r50_L96_H96_B24_per_channel_balanced",
        "train_seeds": TRAIN_SEEDS,
        "test_mask_seeds": TEST_MASK_SEEDS,
        "datasets": DATASETS,
        "baselines": BASELINES,
        "mask_definition": (
            "Every channel has exactly 48 missing observations in its 96-step history. "
            "P scatters these observations; B arranges them into two non-overlapping 24-step blocks."
        ),
        "jobs": jobs,
    }
    atomic_json_dump(payload, args.out)
    print(f"wrote {len(jobs)} training jobs and {sum(len(job['evaluations']) for job in jobs)} evaluations to {args.out}")


if __name__ == "__main__":
    main()
