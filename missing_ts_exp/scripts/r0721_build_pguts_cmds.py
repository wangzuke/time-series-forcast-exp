#!/usr/bin/env python3
"""Generate command files for the 0721 P-GUTS / HD-PGUTS experiment plan."""
from __future__ import annotations

import argparse
import os
import pathlib


ROOT = pathlib.Path("/data/wangzuke/time-series-forecast-exp")
SCRIPT_DIR = ROOT / "missing_ts_exp" / "scripts"
ENTRY = f"{os.environ.get('R0721_PYTHON_CMD', 'python')} -m missing_ts_exp.src.training.run_pguts_hdpguts"
BATCH_SIZE = int(os.environ.get("R0721_BATCH_SIZE", "512"))
NUM_WORKERS = int(os.environ.get("R0721_NUM_WORKERS", "2"))


def cmd(
    dataset: str,
    mask_type: str,
    rate: float,
    horizon: int,
    pooling: str,
    seed: int,
    model: str,
    variant: str,
    epochs: int,
    extra: str = "",
    save_predictions: bool = False,
) -> str:
    pred_flag = "--save_predictions" if save_predictions else ""
    return (
        f"{ENTRY} --dataset {dataset} --mask_type {mask_type} --missing_rate {rate:.2f} "
        f"--T_in 24 --T_out {horizon} --pooling_factors {pooling} "
        f"--model {model} --variant {variant} --seed {seed} --batch_size {BATCH_SIZE} "
        f"--num_workers {NUM_WORKERS} "
        f"--epochs {epochs} --patience 20 {pred_flag} {extra}"
    ).strip()


def write_lines(path: pathlib.Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path} rows={len(lines)}")


def build_smoke(epochs: int) -> list[str]:
    lines = []
    for pooling in ("3", "3,6"):
        for dataset in ("Metr", "PEMS"):
            for mask_type in ("block_t", "block_st"):
                lines.append(
                    cmd(
                        dataset,
                        mask_type,
                        0.70,
                        24,
                        pooling,
                        1,
                        "pgutsf",
                        "pguts",
                        epochs,
                        "--max_train_batches 5 --max_eval_batches 2",
                        save_predictions=True,
                    )
                )
    return lines


def build_phase2(epochs: int) -> list[str]:
    lines = []
    conditions = [("point", 0.50), ("point", 0.70)]
    conditions += [("block_t", r) for r in (0.50, 0.70, 0.90)]
    conditions += [("block_st", r) for r in (0.50, 0.70, 0.90)]
    for horizon in (24, 12):
        for mask_type, rate in conditions:
            for dataset in ("Metr", "PEMS"):
                for pooling in ("3", "3,6"):
                    save_pred = mask_type in {"block_t", "block_st"} and rate in {0.70, 0.90} and horizon == 24
                    lines.append(
                        cmd(
                            dataset,
                            mask_type,
                            rate,
                            horizon,
                            pooling,
                            1,
                            "pgutsf",
                            "pguts",
                            epochs,
                            save_predictions=save_pred,
                        )
                    )

    for mask_type in ("block_st", "block_t"):
        for rate in (0.70, 0.90):
            for dataset in ("Metr", "PEMS"):
                for pooling in ("3", "3,6"):
                    for seed in (2, 3):
                        lines.append(
                            cmd(
                                dataset,
                                mask_type,
                                rate,
                                24,
                                pooling,
                                seed,
                                "pgutsf",
                                "pguts",
                                epochs,
                                save_predictions=True,
                            )
                        )
    return lines


def build_phase3(epochs: int) -> list[str]:
    lines = []
    for mask_type in ("block_t", "block_st"):
        for rate in (0.70, 0.90):
            for dataset in ("Metr", "PEMS"):
                for variant in ("no_graph_coarsening", "no_adaptive_fusion", "full"):
                    for seed in (1, 2, 3):
                        lines.append(
                            cmd(
                                dataset,
                                mask_type,
                                rate,
                                24,
                                "3,6",
                                seed,
                                "hd_pguts",
                                variant,
                                epochs,
                                save_predictions=True,
                            )
                        )
    return lines


def build_phase3_no_graph_rerun(epochs: int) -> list[str]:
    lines = []
    for mask_type in ("block_t", "block_st"):
        for rate in (0.70, 0.90):
            for dataset in ("Metr", "PEMS"):
                for seed in (1, 2, 3):
                    lines.append(
                        cmd(
                            dataset,
                            mask_type,
                            rate,
                            24,
                            "3,6",
                            seed,
                            "hd_pguts",
                            "no_graph_coarsening",
                            epochs,
                            save_predictions=True,
                        )
                    )
    return lines


def build_phase4(epochs: int) -> list[str]:
    lines = []
    for model, variant in (("pgutsf", "pguts"), ("hd_pguts", "full")):
        for dataset in ("Metr", "PEMS"):
            for rate in (0.70, 0.90):
                for seed in (1, 2, 3):
                    lines.append(
                        cmd(
                            dataset,
                            "block_st",
                            rate,
                            12,
                            "3,6",
                            seed,
                            model,
                            variant,
                            epochs,
                            save_predictions=True,
                        )
                    )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke_epochs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    write_lines(SCRIPT_DIR / "r0721_pguts_smoke_cmds.txt", build_smoke(args.smoke_epochs))
    write_lines(SCRIPT_DIR / "r0721_pguts_phase2_cmds.txt", build_phase2(args.epochs))
    write_lines(SCRIPT_DIR / "r0721_hdpguts_phase3_cmds.txt", build_phase3(args.epochs))
    write_lines(
        SCRIPT_DIR / "r0721_hdpguts_no_graph_coarsening_rerun_cmds.txt",
        build_phase3_no_graph_rerun(args.epochs),
    )
    write_lines(SCRIPT_DIR / "r0721_pguts_phase4_cmds.txt", build_phase4(args.epochs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
