#!/usr/bin/env python3
"""Prepare official P-GUTS configs for METR-LA functional coarse assignments."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "missing_ts_exp/results/0818_metrla_functional_cg"
METHODS = ["raw_corr", "residual_corr", "lag_corr", "random"]


def replace_or_append(text: str, key: str, value: str) -> tuple[str, bool]:
    pattern = re.compile(rf"^({re.escape(key)}\s*:\s*).*$", re.MULTILINE)
    if pattern.search(text):
        return pattern.sub(rf"\1{value}", text), True
    return text + f"\n{key}: {value}\n", False


def prepare_one(base_text: str, out_path: Path, assignment_path: Path) -> None:
    text = base_text
    text, _ = replace_or_append(text, "graph_variant", "full_plus_coarse")
    text, matched_path = replace_or_append(
        text, "coarse_assignment_path", str(assignment_path)
    )
    # Some local P-GUTS patches used a shorter key during early experiments.
    if "coarse_assignment:" in text and not matched_path:
        text, _ = replace_or_append(text, "coarse_assignment", str(assignment_path))
    out_path.write_text(text, encoding="utf-8")
    print(f"wrote {out_path}")


def build(args: argparse.Namespace) -> None:
    config_dir = args.pguts_dir / "config/imputation/r0723"
    base_path = config_dir / "la_cgdist_pguts_36.yaml"
    if not base_path.exists():
        raise FileNotFoundError(
            f"Missing base config {base_path}. Run this in the official P-GUTS checkout "
            "that was used for 0723, or copy the r0723 configs first."
        )
    base_text = base_path.read_text(encoding="utf-8")

    out_dir = args.pguts_dir / "config/imputation/r0818"
    out_dir.mkdir(parents=True, exist_ok=True)

    for method in args.methods:
        assignment_path = (
            args.assignment_dir / f"la_{method}_cf{args.coarse_factor}.npy"
        )
        out_path = out_dir / f"la_func_{method}_pguts_36.yaml"
        prepare_one(base_text, out_path, assignment_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pguts-dir",
        type=Path,
        default=ROOT / "external_repro/pguts",
        help="Official P-GUTS checkout with the 0723 coarse-graph patch.",
    )
    parser.add_argument(
        "--assignment-dir",
        type=Path,
        default=DEFAULT_RESULTS / "coarse_assignments",
    )
    parser.add_argument("--coarse-factor", type=int, default=4)
    parser.add_argument("--methods", nargs="+", default=METHODS)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
