#!/usr/bin/env python3
"""Build command files for the 0723 official P-GUTS coarse-graph imputation runs.

The training command files intentionally exclude Table 3 robustness settings.
In the P-GUTS paper, robustness is evaluated by reusing a trained [3,6]
checkpoint and replacing the test eval mask at inference time.
"""
from __future__ import annotations

import argparse
import pathlib


ROOT = pathlib.Path("/data/wangzuke/time-series-forecast-exp")
SCRIPT_DIR = ROOT / "missing_ts_exp" / "scripts"


DATASETS = {
    "la": "la_block",
    "bay": "bay_block",
}

DEFAULT_BATCH_SIZES = {
    "la": 128,
    "bay": 256,
}

DEFAULT_BATCHES_EPOCH = {
    "la": 19,
    "bay": 10,
}


def cmd(config: str, dataset: str, seed: int, extra: list[str] | None = None) -> str:
    parts = [
        "python",
        "-m",
        "experiments.run_imputation",
        "--config",
        config,
        "--dataset-name",
        dataset,
        "--seed",
        str(seed),
    ]
    if extra:
        parts.extend(extra)
    return " ".join(parts)


def write_lines(path: pathlib.Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n")
    print(f"{path}: {len(lines)} commands")


def build(args: argparse.Namespace) -> None:
    smoke = []
    reproduce = []
    coarse = []

    for ds_key, dataset_name in DATASETS.items():
        batch_size = args.batch_size[ds_key]
        batches_epoch = args.batches_epoch[ds_key]
        smoke_extra = [
            "--epochs",
            str(args.smoke_epochs),
            "--batches-epoch",
            str(args.smoke_batches_epoch),
            "--patience",
            str(args.smoke_patience),
            "--batch-size",
            str(batch_size),
        ]

        smoke.append(
            cmd(
                f"imputation/r0723/{ds_key}_pguts_36.yaml",
                dataset_name,
                args.seeds[0],
                smoke_extra,
            )
        )
        smoke.append(
            cmd(
                f"imputation/r0723/{ds_key}_cg_pguts_36.yaml",
                dataset_name,
                args.seeds[0],
                smoke_extra,
            )
        )

        for seed in args.seeds:
            # Paper reproduction, traffic subset:
            # Keep only the paper's best traffic setting [3,6] for this round.
            reproduce.append(
                cmd(
                    f"imputation/r0723/{ds_key}_pguts_36.yaml",
                    dataset_name,
                    seed,
                    [
                        "--batch-size",
                        str(batch_size),
                        "--batches-epoch",
                        str(batches_epoch),
                    ],
                )
            )

            # New module experiment. Baseline P-GUTS [3,6] is already covered
            # by the reproduction commands above, so avoid duplicate runs here.
            coarse.append(
                cmd(
                    f"imputation/r0723/{ds_key}_cg_pguts_36.yaml",
                    dataset_name,
                    seed,
                    [
                        "--batch-size",
                        str(batch_size),
                        "--batches-epoch",
                        str(batches_epoch),
                    ],
                )
            )

    write_lines(SCRIPT_DIR / "r0723_official_pguts_cg_smoke_cmds.txt", smoke)
    write_lines(SCRIPT_DIR / "r0723_official_pguts_reproduce_cmds.txt", reproduce)
    write_lines(SCRIPT_DIR / "r0723_official_pguts_cg_coarse_cmds.txt", coarse)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-epochs", type=int, default=5)
    parser.add_argument("--smoke-batches-epoch", type=int, default=2)
    parser.add_argument("--smoke-patience", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--la-batch-size", type=int, default=DEFAULT_BATCH_SIZES["la"])
    parser.add_argument("--bay-batch-size", type=int, default=DEFAULT_BATCH_SIZES["bay"])
    parser.add_argument(
        "--la-batches-epoch", type=int, default=DEFAULT_BATCHES_EPOCH["la"]
    )
    parser.add_argument(
        "--bay-batches-epoch", type=int, default=DEFAULT_BATCHES_EPOCH["bay"]
    )
    args = parser.parse_args()
    args.batch_size = {
        "la": args.la_batch_size,
        "bay": args.bay_batch_size,
    }
    args.batches_epoch = {
        "la": args.la_batches_epoch,
        "bay": args.bay_batches_epoch,
    }
    return args


if __name__ == "__main__":
    build(parse_args())
