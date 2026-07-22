#!/usr/bin/env python3
"""Preflight checks for the 0721 P-GUTS / HD-PGUTS runtime environment."""
from __future__ import annotations

import argparse
import importlib
import json
import pathlib
import sys


ROOT = pathlib.Path("/data/wangzuke/time-series-forecast-exp")
MASK_DIR = ROOT / "dataset" / "0721_missing_masks"
DATA_PATHS = [
    ROOT / "dataset" / "metr_la" / "metr_la.h5",
    ROOT / "dataset" / "pems_bay" / "pems_bay.h5",
]


def import_required(name: str):
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        raise RuntimeError(f"missing required module: {name} ({exc})") from exc
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_h5_read", action="store_true")
    parser.add_argument("--require_cuda", action="store_true")
    args = parser.parse_args()

    report: dict[str, object] = {"python": sys.executable, "checks": {}}
    numpy = import_required("numpy")
    pandas = import_required("pandas")
    torch = import_required("torch")
    tables = import_required("tables")
    report["checks"].update({
        "numpy": getattr(numpy, "__version__", "unknown"),
        "pandas": getattr(pandas, "__version__", "unknown"),
        "torch": getattr(torch, "__version__", "unknown"),
        "tables": getattr(tables, "__version__", "unknown"),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    })

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but torch.cuda.is_available() is false")

    manifest = MASK_DIR / "manifest.csv"
    if not manifest.exists():
        raise RuntimeError(f"missing mask manifest: {manifest}")
    report["checks"]["manifest"] = str(manifest)

    if not args.skip_h5_read:
        h5_shapes = {}
        for path in DATA_PATHS:
            if not path.exists():
                raise RuntimeError(f"missing HDF5 data file: {path}")
            df = pandas.read_hdf(path)
            h5_shapes[str(path)] = list(df.shape)
        report["checks"]["h5_shapes"] = h5_shapes

    print(json.dumps(report, indent=2, ensure_ascii=True))
    print("[r0721 env] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
