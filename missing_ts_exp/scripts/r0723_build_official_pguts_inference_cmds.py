#!/usr/bin/env python3
"""Build Table 3 inference commands from finished [3,6] checkpoints.

The P-GUTS robustness table evaluates one trained checkpoint under multiple
test masks. It must not retrain separate models for each failure probability.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import yaml


ROOT = pathlib.Path("/data/wangzuke/time-series-forecast-exp")
LOG_ROOT = ROOT / "external_repro" / "pguts" / "log"
SCRIPT_DIR = ROOT / "missing_ts_exp" / "scripts"

DATASETS = {
    "la": "la_block",
    "bay": "bay_block",
}

EXPECTED_BATCHES_EPOCH = {
    "la_block": 19,
    "bay_block": 10,
}

EXPECTED_BATCH_SIZE = {
    "la_block": 128,
    "bay_block": 256,
}

GRAPH_VARIANTS = {
    "pguts": "full_only",
    "cg_pguts": "full_plus_coarse",
}

P_FAULTS = [0.05, 0.10, 0.15]
MASK_SEEDS = [6043, 2043, 3043, 4043, 5043]


def read_yaml(path: pathlib.Path) -> dict:
    with path.open("r") as fp:
        return yaml.safe_load(fp)


def config_matches(cfg: dict, dataset_name: str, model_key: str, seed: int) -> bool:
    ds_key = dataset_name.split("_", 1)[0]
    expected_config = (
        f"{ds_key}_cg_pguts_36" if model_key == "cg_pguts" else f"{ds_key}_pguts_36"
    )
    return (
        cfg.get("dataset_name") == dataset_name
        and cfg.get("seed") == seed
        and cfg.get("graph_variant", "full_only") == GRAPH_VARIANTS[model_key]
        and cfg.get("factor_t") == [3, 6]
        and cfg.get("p_fault") == 0.0015
        and cfg.get("p_noise") == 0.05
        and cfg.get("batch_size") == EXPECTED_BATCH_SIZE[dataset_name]
        and cfg.get("batches_epoch") == EXPECTED_BATCHES_EPOCH[dataset_name]
        and pathlib.Path(cfg.get("config", "")).stem == expected_config
    )


def find_finished_exp(dataset_name: str, model_key: str, seed: int) -> pathlib.Path | None:
    model_dir = LOG_ROOT / dataset_name / "PGUTS"
    if not model_dir.exists():
        return None

    candidates: list[pathlib.Path] = []
    for cfg_path in model_dir.glob("*/config.yaml"):
        try:
            cfg = read_yaml(cfg_path)
        except Exception:
            continue
        exp_dir = cfg_path.parent
        if not (exp_dir / "output.pt").exists():
            continue
        if config_matches(cfg, dataset_name, model_key, seed):
            candidates.append(exp_dir)

    if not candidates:
        return None
    return sorted(candidates, key=lambda path: path.stat().st_mtime)[-1]


def infer_cmd(dataset_name: str, exp_dir: pathlib.Path, p_fault: float, batch_size: int) -> str:
    mask_seed = ",".join(str(seed) for seed in MASK_SEEDS)
    return (
        "python -m experiments.run_inference "
        "--config inference.yaml "
        "--model-name PGUTS "
        f"--dataset-name {dataset_name} "
        f"--exp-name {exp_dir.name} "
        f"--p-fault {p_fault:g} "
        "--p-noise 0 "
        f"--test-mask-seed {mask_seed} "
        f"--batch-size {batch_size}"
    )


def build(args: argparse.Namespace) -> int:
    lines: list[str] = []
    missing: list[str] = []

    for _, dataset_name in DATASETS.items():
        for seed in args.seeds:
            for model_key in args.models:
                exp_dir = find_finished_exp(dataset_name, model_key, seed)
                if exp_dir is None:
                    missing.append(f"{dataset_name} {model_key} seed={seed}")
                    continue
                for p_fault in args.p_faults:
                    lines.append(
                        infer_cmd(
                            dataset_name=dataset_name,
                            exp_dir=exp_dir,
                            p_fault=p_fault,
                            batch_size=args.batch_size[dataset_name],
                        )
                    )

    out_path = SCRIPT_DIR / "r0723_official_pguts_cg_table3_inference_cmds.txt"
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""))
    print(f"{out_path}: {len(lines)} commands")

    missing_path = SCRIPT_DIR / "r0723_official_pguts_cg_table3_missing_checkpoints.txt"
    if missing:
        missing_path.write_text("\n".join(missing) + "\n")
        print(f"{missing_path}: {len(missing)} missing checkpoints")
    elif missing_path.exists():
        missing_path.unlink()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    parser.add_argument(
        "--models",
        choices=sorted(GRAPH_VARIANTS),
        nargs="+",
        default=["pguts", "cg_pguts"],
    )
    parser.add_argument("--p-faults", type=float, nargs="+", default=P_FAULTS)
    parser.add_argument("--la-batch-size", type=int, default=64)
    parser.add_argument("--bay-batch-size", type=int, default=48)
    args = parser.parse_args()
    args.batch_size = {
        "la_block": args.la_batch_size,
        "bay_block": args.bay_batch_size,
    }
    return args


if __name__ == "__main__":
    sys.exit(build(parse_args()))
