#!/usr/bin/env python3
"""Create the isolated S4M depth-selection manifest for the r=50% protocol."""
from __future__ import annotations

import argparse
import json
import pathlib

from protocol import EXP_ROOT, atomic_json_dump


SUITE = "s4m_depth_selection_r50_l96_h96_official_sgd"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path,
                        default=EXP_ROOT / "configs" / f"{SUITE}.json")
    args = parser.parse_args()
    jobs = []
    for dataset in ("ETTh1", "Weather", "Electricity"):
        for e_layers in (2, 4, 8):
            jobs.append({
                "id": f"S4MDepth_{dataset}_L96_H96_P_r50_e{e_layers}_s2024",
                "baseline": "S4M", "dataset": dataset, "train_family": "P",
                "seq_len": 96, "pred_len": 96, "missing_rate": 0.5,
                "mask_balance": "per_channel", "block_len": 24,
                "train_seed": 2024, "mask_seed": 9101, "e_layers": e_layers,
            })
    atomic_json_dump({
        "suite": SUITE,
        "purpose": "Validation-only S4M depth selection with the released S4M training entrypoint's SGD; no test-set metric is used for selection.",
        "optimizer": {"name": "SGD", "learning_rate_by_dataset": {"ETTh1": 0.005, "Weather": 0.001, "Electricity": 0.01},
                      "momentum": 0.9, "weight_decay": 1e-5, "source": "released experiments/exp_pretrain1.py"},
        "selection_rule": {
            "primary": "lowest P-family validation objective among candidates passing sensitivity screens",
            "history_permutation_relative_prediction_l1_min": 0.001,
            "p_to_b_relative_prediction_l1_min": 0.001,
        },
        "jobs": jobs,
    }, args.output)


if __name__ == "__main__":
    main()
