#!/usr/bin/env python3
"""Build the isolated Traffic matrix and strict-mask Electricity S4M jobs."""
from __future__ import annotations

import argparse
import pathlib

from protocol import EXP_ROOT, atomic_json_dump


SUITE = "s4m_traffic_electricity_stability_1115"
TRAIN_SEEDS = (2024, 2025)
TEST_MASK_SEEDS = (9101, 9102)
TRAFFIC_MODELS = ("S4M", "ChannelTokenFormer", "PatchTST", "DLinear", "iTransformer")
ABLATIONS = ("no_mask_stream", "no_prototype_retrieval")


def make_job(dataset: str, baseline: str, family: str, seed: int, *, ablation: str = "none") -> dict:
    suffix = "" if ablation == "none" else f"_{ablation}"
    phase = "traffic" if dataset == "Traffic" else ("electricity_reference" if ablation == "none" else "electricity_ablation")
    identity = f"{dataset}_{baseline}{suffix}_{family}_b24_s{seed}"
    return {
        "id": identity,
        "phase": phase,
        "dataset": dataset,
        "baseline": baseline,
        "ablation": ablation,
        "seq_len": 96,
        "pred_len": 96,
        "missing_rate": 0.5,
        "mask_balance": "per_channel",
        "train_family": family,
        "train_block_len": 24,
        "train_seed": seed,
        "test_mask_seeds": list(TEST_MASK_SEEDS),
        "evaluations": [
            {"test_family": test_family, "block_len": 24, "test_mask_seed": test_seed}
            for test_family in ("P", "B")
            for test_seed in TEST_MASK_SEEDS
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path,
                        default=EXP_ROOT / "configs" / f"{SUITE}.json")
    args = parser.parse_args()
    jobs = []
    for model in TRAFFIC_MODELS:
        for family in ("P", "B"):
            for seed in TRAIN_SEEDS:
                jobs.append(make_job("Traffic", model, family, seed))
    for family in ("P", "B"):
        for seed in TRAIN_SEEDS:
            jobs.append(make_job("Electricity", "S4M", family, seed))
            for ablation in ABLATIONS:
                jobs.append(make_job("Electricity", "S4M", family, seed, ablation=ablation))
    atomic_json_dump({
        "suite": SUITE,
        "protocol": "r50_per_channel_L96_H96_B24_two_train_seeds_two_mask_seeds_strict_separated_blocks",
        "train_seeds": list(TRAIN_SEEDS),
        "test_mask_seeds": list(TEST_MASK_SEEDS),
        "traffic_models": list(TRAFFIC_MODELS),
        "electricity_reference": "S4M strict-mask rerun, four checkpoints",
        "electricity_ablations": list(ABLATIONS),
        "jobs": jobs,
    }, args.output)
    print(f"wrote {len(jobs)} training jobs to {args.output}")


if __name__ == "__main__":
    main()
