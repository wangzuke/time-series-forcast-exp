#!/usr/bin/env python3
"""Build R0723 graph-aware coarse P-GUTS training commands."""

from __future__ import annotations

from pathlib import Path


ROOT = Path("/data/wangzuke/time-series-forecast-exp")
SCRIPT_DIR = ROOT / "missing_ts_exp/scripts"

DATASETS = {
    "la": {
        "dataset_name": "la_block",
        "batch_size": 128,
        "batches_epoch": 19,
    },
    "bay": {
        "dataset_name": "bay_block",
        "batch_size": 256,
        "batches_epoch": 10,
    },
}


def build_command(ds_key: str, seed: int) -> str:
    spec = DATASETS[ds_key]
    return " ".join(
        [
            "python",
            "-m",
            "experiments.run_imputation",
            "--config",
            f"imputation/r0723/{ds_key}_cgdist_pguts_36.yaml",
            "--dataset-name",
            spec["dataset_name"],
            "--seed",
            str(seed),
            "--batch-size",
            str(spec["batch_size"]),
            "--batches-epoch",
            str(spec["batches_epoch"]),
        ]
    )


def main() -> None:
    commands = []
    for ds_key in ("la", "bay"):
        for seed in (1, 2):
            commands.append(build_command(ds_key, seed))
    out_path = SCRIPT_DIR / "r0723_official_pguts_graph_aware_cg_cmds.txt"
    out_path.write_text("\n".join(commands) + "\n", encoding="utf-8")
    print(f"{out_path}: {len(commands)} commands")


if __name__ == "__main__":
    main()
