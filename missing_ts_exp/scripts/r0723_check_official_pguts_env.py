#!/usr/bin/env python3
"""Preflight checks for the official P-GUTS coarse-graph imputation runs."""
from __future__ import annotations

import importlib
import os
import pathlib
import sys


ROOT = pathlib.Path("/data/wangzuke/time-series-forecast-exp")
PGUTS_DIR = ROOT / "external_repro" / "pguts"


def check_import(module: str) -> bool:
    try:
        imported = importlib.import_module(module)
    except Exception as exc:
        print(f"FAIL import {module}: {type(exc).__name__}: {exc}")
        return False
    print(f"OK import {module} {getattr(imported, '__version__', '')}".strip())
    return True


def main() -> int:
    ok = True
    tsl_data_dir = os.environ.get("R0723_TSL_DATA_DIR")
    if tsl_data_dir:
        from tsl import config

        pathlib.Path(tsl_data_dir).mkdir(parents=True, exist_ok=True)
        config.data_dir = tsl_data_dir
        print(f"OK TSL data_dir {config.data_dir}")

    for path in (
        PGUTS_DIR / "experiments" / "run_imputation.py",
        PGUTS_DIR / "code" / "models" / "pguts.py",
        PGUTS_DIR / "config" / "imputation" / "r0723" / "la_pguts_36.yaml",
    ):
        if path.exists():
            print(f"OK file {path}")
        else:
            print(f"FAIL missing file {path}")
            ok = False

    for module in (
        "torch",
        "pytorch_lightning",
        "tsl",
        "torch_geometric",
        "torch_scatter",
        "torch_sparse",
        "wandb",
    ):
        ok = check_import(module) and ok

    if tsl_data_dir:
        try:
            from tsl.datasets import MetrLA, PemsBay

            for cls in (MetrLA, PemsBay):
                dataset = cls()
                print(
                    f"OK dataset {cls.__name__} "
                    f"length={len(dataset)} nodes={dataset.n_nodes}"
                )
        except Exception as exc:
            print(f"FAIL dataset load: {type(exc).__name__}: {exc}")
            ok = False

    if not ok:
        print(
            "Official P-GUTS dependencies are incomplete. "
            "Create the environment from external_repro/pguts/environment.yml, "
            "then install torch-scatter and torch-sparse from the PyG wheel index "
            "matching that torch/CUDA version."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
