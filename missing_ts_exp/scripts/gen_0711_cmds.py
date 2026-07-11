#!/usr/bin/env python3
"""Generate 0711 experiment commands.

Phase 0: checkpoint-enabled diagnostic reruns for soft/fuse.
Phase 1: reliability-gated fuse variants (sgate/ggate/mgate).

All commands are meant to run from the missing_ts_exp directory.
"""
from __future__ import annotations

import os

RESULTS_DIR = "results"
CKPT_DIR = "results/checkpoints/r0711"
SEEDS = [2024, 2025]
EPOCHS = 20
PATIENCE = 5
BATCH_SIZE = 32
LR = "1e-3"
PRED_LEN = 96


def result_exists(tag: str) -> bool:
    return os.path.exists(os.path.join(RESULTS_DIR, f"{tag}.json"))


def cmd(tag: str, dataset: str, missing_type: str, missing_rate: float, seed: int, variant: str) -> str:
    return (
        "python -m src.training.run_forecast "
        f"--dataset {dataset} --method misstsm --predictor iTransformer "
        f"--impute none --missing_type {missing_type} --missing_rate {missing_rate} "
        f"--seq_len 96 --pred_len {PRED_LEN} --seed {seed} "
        f"--epochs {EPOCHS} --batch_size {BATCH_SIZE} --lr {LR} --patience {PATIENCE} "
        f"--tag {tag} --out_dir {RESULTS_DIR} --misstsm_variant {variant} "
        f"--save_checkpoint_dir {CKPT_DIR}"
    )


def rate_label(rate: float) -> int:
    return int(round(rate * 100))


def gen_phase0() -> list[str]:
    """Diagnostic reruns from docs/实验计划0711.md §3.3."""
    specs = []
    for rate in [0.5, 0.7]:
        for variant, short in [("grouped_q4_soft", "gq4soft"), ("grouped_q4_fuse", "gq4fuse")]:
            specs.append(("Weather", "continuous_segment", rate, variant, short))
    specs.append(("Weather", "random_point", 0.5, "grouped_q4_fuse", "gq4fuse"))
    for rate in [0.3, 0.5]:
        for variant, short in [("grouped_q4_soft", "gq4soft"), ("grouped_q4_fuse", "gq4fuse")]:
            specs.append(("Traffic", "random_point", rate, variant, short))
    for variant, short in [("grouped_q4_soft", "gq4soft"), ("grouped_q4_fuse", "gq4fuse")]:
        specs.append(("Traffic", "continuous_segment", 0.7, variant, short))
    specs.append(("Electricity", "random_point", 0.5, "grouped_q4_fuse", "gq4fuse"))

    cmds = []
    for dataset, missing_type, rate, variant, short in specs:
        for seed in SEEDS:
            tag = (
                f"r0711_P0diag__misstsm_{short}__{dataset}__"
                f"{missing_type}_{rate_label(rate)}__h{PRED_LEN}_s{seed}"
            )
            if not result_exists(tag):
                cmds.append(cmd(tag, dataset, missing_type, rate, seed, variant))
    return cmds


def gen_phase1() -> list[str]:
    """Reliability-gated fuse main experiments from docs/实验计划0711.md §4.3."""
    conditions = [
        ("Weather", "continuous_segment", 0.3),
        ("Weather", "continuous_segment", 0.5),
        ("Weather", "continuous_segment", 0.7),
        ("Weather", "random_point", 0.5),
        ("Traffic", "random_point", 0.3),
        ("Traffic", "random_point", 0.5),
        ("Traffic", "random_point", 0.7),
        ("Traffic", "continuous_segment", 0.7),
        ("Electricity", "random_point", 0.5),
        ("Electricity", "continuous_segment", 0.5),
    ]
    variants = [
        ("grouped_q4_fuse_sgate", "fuse_sgate"),
        ("grouped_q4_fuse_ggate", "fuse_ggate"),
        ("grouped_q4_fuse_mgate", "fuse_mgate"),
    ]
    cmds = []
    for dataset, missing_type, rate in conditions:
        for variant, short in variants:
            for seed in SEEDS:
                tag = (
                    f"r0711_P1__misstsm_{short}__{dataset}__"
                    f"{missing_type}_{rate_label(rate)}__h{PRED_LEN}_s{seed}"
                )
                if not result_exists(tag):
                    cmds.append(cmd(tag, dataset, missing_type, rate, seed, variant))
    return cmds


def write(path: str, cmds: list[str]):
    with open(path, "w") as f:
        if cmds:
            f.write("\n".join(cmds) + "\n")
    print(f"wrote {len(cmds)} commands to {path}")


def main():
    phase0 = gen_phase0()
    phase1 = gen_phase1()
    write("scripts/r0711_phase0_cmds.txt", phase0)
    write("scripts/r0711_phase1_cmds.txt", phase1)
    write("scripts/r0711_cmds.txt", phase0 + phase1)


if __name__ == "__main__":
    main()
