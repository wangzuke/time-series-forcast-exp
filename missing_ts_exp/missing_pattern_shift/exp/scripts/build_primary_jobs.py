#!/usr/bin/env python3
"""Create the resumable first-round 2x2 job manifest.

One job trains exactly one (baseline, data, H, train mask, train seed) model
and then evaluates every appropriate fixed test-mask seed.  This avoids
retraining a model five times merely to change an evaluation mask.
"""
from __future__ import annotations

import argparse
import json
import pathlib

from protocol import EXP_ROOT, atomic_json_dump


# Keep the original enumeration universe so two-seed job IDs remain identical
# to the initial five-seed manifest.  This lets completed 2024/2025 artifacts
# be resumed safely while preserving the already-collected 2026--2028 files as
# an archived exploratory record rather than silently reusing them.
ALL_TRAIN_SEEDS = [2024, 2025, 2026, 2027, 2028]
DEFAULT_TRAIN_SEEDS = [2024, 2025]
DEFAULT_TEST_MASK_SEEDS = [9101, 9102]
ALL_HORIZONS = [24, 96]
BLOCK_LENS = [12, 24, 48]
PAPER_ALIGNED_DATASETS = {
    # Both data sets appear in BiTGraph's ICLR'24 evaluation.
    "BiTGraph": {"ETTh1", "Electricity"},
    # CTF's released regular benchmark commands cover ETTh1 and Weather.
    "ChannelTokenFormer": {"ETTh1", "Weather"},
    "S4M": {"ETTh1", "Weather", "Electricity"},
}


def make_job(index: int, baseline: str, dataset: str, horizon: int, train_family: str, block_len: int, train_seed: int,
             test_mask_seeds: list[int], test_block_lens: list[int]) -> dict:
    evaluations: list[dict] = []
    if train_family == "P":
        for test_seed in test_mask_seeds:
            evaluations.append({"test_family": "P", "block_len": 24, "test_mask_seed": test_seed})
        for test_block_len in test_block_lens:
            for test_seed in test_mask_seeds:
                evaluations.append({"test_family": "B", "block_len": test_block_len, "test_mask_seed": test_seed})
    else:
        # B->B is evaluated at the same block length seen during training.
        # B->P remains separately identifiable by the training block length.
        for test_seed in test_mask_seeds:
            evaluations.append({"test_family": "P", "block_len": block_len, "test_mask_seed": test_seed})
            evaluations.append({"test_family": "B", "block_len": block_len, "test_mask_seed": test_seed})
    return {
        "id": f"{index:04d}_{baseline}_{dataset}_H{horizon}_{train_family}_b{block_len}_s{train_seed}",
        "baseline": baseline, "dataset": dataset, "pred_len": horizon,
        "train_family": train_family, "train_block_len": block_len,
        "train_seed": train_seed, "evaluations": evaluations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["ETTh1", "Weather", "Electricity"])
    parser.add_argument("--baselines", nargs="+", default=["S4M", "BiTGraph", "ChannelTokenFormer"])
    parser.add_argument("--include-ctf-electricity", action="store_true")
    parser.add_argument("--paper-aligned", action="store_true",
                        help="retain only data sets supported by each baseline's paper/released commands")
    parser.add_argument("--train-seeds", nargs="+", type=int, default=DEFAULT_TRAIN_SEEDS)
    parser.add_argument("--test-mask-seeds", nargs="+", type=int, default=DEFAULT_TEST_MASK_SEEDS)
    parser.add_argument("--horizons", nargs="+", type=int, default=ALL_HORIZONS,
                        help="forecast horizons to retain; IDs preserve the full enumeration universe")
    parser.add_argument("--block-lens", nargs="+", type=int, default=BLOCK_LENS,
                        help="block lengths to retain for B-training and B-testing")
    parser.add_argument("--out", type=pathlib.Path, default=EXP_ROOT / "configs" / "primary_jobs.json")
    args = parser.parse_args()
    train_seeds = list(dict.fromkeys(args.train_seeds))
    unsupported = sorted(set(train_seeds) - set(ALL_TRAIN_SEEDS))
    if unsupported:
        raise SystemExit(f"unsupported train seeds: {unsupported}; choose from {ALL_TRAIN_SEEDS}")
    test_mask_seeds = list(dict.fromkeys(args.test_mask_seeds))
    if not test_mask_seeds:
        raise SystemExit("at least one test-mask seed is required")
    horizons = list(dict.fromkeys(args.horizons))
    if unsupported := sorted(set(horizons) - set(ALL_HORIZONS)):
        raise SystemExit(f"unsupported horizons: {unsupported}; choose from {ALL_HORIZONS}")
    block_lens = list(dict.fromkeys(args.block_lens))
    if unsupported := sorted(set(block_lens) - set(BLOCK_LENS)):
        raise SystemExit(f"unsupported block lengths: {unsupported}; choose from {BLOCK_LENS}")
    if not block_lens:
        raise SystemExit("at least one block length is required")
    jobs = []
    index = 0
    for baseline in args.baselines:
        for dataset in args.datasets:
            if args.paper_aligned and dataset not in PAPER_ALIGNED_DATASETS[baseline]:
                continue
            if baseline == "ChannelTokenFormer" and dataset == "Electricity" and not args.include_ctf_electricity:
                continue
            for horizon in ALL_HORIZONS:
                for train_seed in ALL_TRAIN_SEEDS:
                    candidates = [
                        make_job(index, baseline, dataset, horizon, "P", 24, train_seed, test_mask_seeds, block_lens),
                        *[make_job(index + offset, baseline, dataset, horizon, "B", block_len, train_seed, test_mask_seeds, block_lens)
                          for offset, block_len in enumerate(BLOCK_LENS, start=1)],
                    ]
                    if train_seed in train_seeds and horizon in horizons:
                        jobs.extend(job for job in candidates if job["train_family"] == "P" or job["train_block_len"] in block_lens)
                    index += len(candidates)
    payload = {
        "protocol": f"main_2x2_P_to_B_r30_L96_{len(train_seeds)}trainx{len(test_mask_seeds)}test",
        "train_seeds": train_seeds, "test_mask_seeds": test_mask_seeds,
        "horizons": horizons, "block_lens": block_lens, "jobs": jobs,
        "note": (
            "Per-baseline paper-aligned data-set coverage is enforced."
            if args.paper_aligned else "CTF/Electricity is excluded by default until its 24GB smoke test succeeds."
        ),
    }
    atomic_json_dump(payload, args.out)
    print(f"wrote {len(jobs)} train jobs and {sum(len(job['evaluations']) for job in jobs)} evaluations to {args.out}")


if __name__ == "__main__":
    main()
