#!/usr/bin/env python3
"""Build METR-LA functional coarse-graph P-GUTS command files."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "missing_ts_exp/scripts"
METHODS = ["raw_corr", "residual_corr", "lag_corr", "random"]
P_FAULTS = [0.05, 0.10, 0.15]
MASK_SEEDS = "6043,2043,3043,4043,5043"


def train_cmd(method: str, seed: int, smoke: bool, args: argparse.Namespace) -> str:
    parts = [
        "python",
        "-m",
        "experiments.run_imputation",
        "--config",
        f"imputation/r0818/la_func_{method}_pguts_36.yaml",
        "--dataset-name",
        "la_block",
        "--seed",
        str(seed),
        "--batch-size",
        str(args.batch_size),
        "--batches-epoch",
        str(args.smoke_batches_epoch if smoke else args.batches_epoch),
    ]
    if smoke:
        parts.extend(["--epochs", str(args.smoke_epochs), "--patience", str(args.smoke_patience)])
    return " ".join(parts)


def table3_cmd(exp_name: str, p_fault: float, batch_size: int) -> str:
    return (
        "python -m experiments.run_inference "
        "--config inference.yaml "
        "--model-name PGUTS "
        "--dataset-name la_block "
        f"--exp-name {exp_name} "
        f"--p-fault {p_fault:g} "
        "--p-noise 0 "
        f"--test-mask-seed {MASK_SEEDS} "
        f"--batch-size {batch_size}"
    )


def write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{path}: {len(lines)} commands")


def build(args: argparse.Namespace) -> None:
    smoke = [train_cmd(method, args.seeds[0], True, args) for method in args.methods]
    train = [
        train_cmd(method, seed, False, args)
        for method in args.methods
        for seed in args.seeds
    ]

    write_lines(SCRIPT_DIR / "r0818_metrla_functional_cg_smoke_cmds.txt", smoke)
    write_lines(SCRIPT_DIR / "r0818_metrla_functional_cg_train_cmds.txt", train)

    if args.exp_names_file:
        exp_names = [
            line.strip()
            for line in args.exp_names_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        table3 = [
            table3_cmd(exp_name, p_fault, args.inference_batch_size)
            for exp_name in exp_names
            for p_fault in args.p_faults
        ]
        write_lines(SCRIPT_DIR / "r0818_metrla_functional_cg_table3_cmds.txt", table3)
    else:
        template = SCRIPT_DIR / "r0818_metrla_functional_cg_table3_exp_names_template.txt"
        template.write_text(
            "\n".join(
                [
                    "# Fill with finished official P-GUTS exp directory names, one per line.",
                    "# Example:",
                    "# 20260818T120000_1_r0818_func_residual_corr_s1",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"{template}: fill after training, then rerun with --exp-names-file")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--batches-epoch", type=int, default=300)
    parser.add_argument("--smoke-epochs", type=int, default=5)
    parser.add_argument("--smoke-batches-epoch", type=int, default=2)
    parser.add_argument("--smoke-patience", type=int, default=2)
    parser.add_argument("--p-faults", type=float, nargs="+", default=P_FAULTS)
    parser.add_argument("--inference-batch-size", type=int, default=64)
    parser.add_argument("--exp-names-file", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
