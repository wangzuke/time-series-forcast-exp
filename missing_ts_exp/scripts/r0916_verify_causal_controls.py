#!/usr/bin/env python3
"""Verify the 20-record, paired causal-control matrix and compute fixed statistics."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats


ROOT = Path("/media/data1/wangzk/time-series-forcast-exp")
RESULT = ROOT / "missing_ts_exp/results/r0916_causal_controls"
OLD = ROOT / "missing_ts_exp/results/r0908_official_baselines/controlled"
STAGE_B = ROOT / "missing_ts_exp/results/r0910_evidence_closure/stage_b_ablation"
OUT = RESULT / "verification_summary.json"
P0 = RESULT / "p0_checks.json"
SEEDS = (2024, 2025, 2026, 2027, 2028)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def path_and_tag(variant: str, seed: int) -> tuple[Path, str]:
    suffix = f"METR-LA_block_t_r50_seed{seed}"
    if variant == "official_base":
        return (OLD if seed <= 2026 else STAGE_B), f"official_base_{suffix}"
    if variant == "full":
        return (OLD if seed <= 2026 else STAGE_B), f"{'mrmsgf' if seed <= 2026 else 'full'}_{suffix}"
    if variant == "feature_only":
        return RESULT, f"official_base_feature_only_{suffix}"
    if variant == "permuted_q":
        return RESULT, f"mrmsgf_permuted_q_{suffix}"
    raise ValueError(variant)


def checkpoint_exists(directory: Path, row: dict) -> bool:
    explicit = row.get("best_checkpoint")
    if explicit:
        return Path(explicit).is_file()
    return any((directory / row["tag"]).glob("*.ckpt"))


def read_row(variant: str, seed: int) -> dict:
    directory, tag = path_and_tag(variant, seed)
    path = directory / f"{tag}.json"
    require(path.is_file(), f"missing result JSON: {path}")
    row = json.loads(path.read_text())
    require(row.get("tag") == tag, f"tag mismatch: {path}")
    expected_model = "official_base" if variant == "official_base" else (
        "official_base_feature_only" if variant == "feature_only" else "mrmsgf"
    )
    require(row.get("model") == expected_model, f"model mismatch: {path}")
    require(row.get("seed") == seed and row.get("epochs") == 100 and row.get("train_batches") == 300,
            f"budget/seed mismatch: {path}")
    require(row.get("official_hdtts_commit") == "46a8717db3802e1684e991ba0a7c0bfe43d22535",
            f"official source mismatch: {path}")
    require(checkpoint_exists(directory, row), f"missing best checkpoint: {path}")
    for section, fields in (("masked_test", ("test_mae", "test_mse")),
                            ("unmasked_test", ("test_mae_unmasked", "test_mse_unmasked"))):
        require(section in row, f"missing {section}: {path}")
        for field in fields:
            require(math.isfinite(float(row[section][field])), f"non-finite {section}.{field}: {path}")
    if variant == "feature_only":
        require(row.get("feature_only_no_dynamic_graph") is True, f"Feature-only architecture audit missing: {path}")
        require(row.get("parameter_count", 0) > 0, f"Feature-only parameter count missing: {path}")
    if variant == "permuted_q":
        metadata = row.get("q_permutation") or {}
        require(row.get("mr_variant") == "full" and row.get("q_mode") == "permuted_pairwise",
                f"Permuted-Q identity mismatch: {path}")
        require(row.get("mask_seed") == seed, f"mask seed mismatch: {path}")
        require(metadata.get("transform") == "Q_perm=P Q P^T", f"permutation transform mismatch: {path}")
        require(metadata.get("seed_components") == ["mask_seed", "split_id", "absolute_window_index"],
                f"permutation seed metadata mismatch: {path}")
        require(metadata.get("window_metadata_input") is True, f"window metadata missing: {path}")
    row["_path"] = str(path)
    return row


def metric_stats(rows: list[dict]) -> dict[str, dict[str, float]]:
    mae = np.asarray([float(row["unmasked_test"]["test_mae_unmasked"]) for row in rows])
    rmse = np.sqrt(np.asarray([float(row["unmasked_test"]["test_mse_unmasked"]) for row in rows]))
    return {
        "mae": {"mean": float(mae.mean()), "std": float(mae.std(ddof=1)), "values": mae.tolist()},
        "rmse": {"mean": float(rmse.mean()), "std": float(rmse.std(ddof=1)), "values": rmse.tolist()},
    }


def paired(values: np.ndarray, full: np.ndarray) -> dict[str, object]:
    delta = values - full
    mean = float(delta.mean())
    std = float(delta.std(ddof=1))
    if std == 0.0:
        ci = [mean, mean]
        p_value = 1.0 if mean == 0.0 else 0.0
    else:
        half = float(stats.t.ppf(0.975, len(delta) - 1) * stats.sem(delta))
        ci = [mean - half, mean + half]
        p_value = float(stats.ttest_rel(values, full).pvalue)
    return {"values": delta.tolist(), "mean": mean, "std": std, "ci95": ci, "p_value": p_value}


def holm(items: dict[str, dict[str, object]]) -> None:
    ordered = sorted(items, key=lambda key: float(items[key]["mae"]["p_value"]))
    running = 0.0
    for rank, key in enumerate(ordered):
        value = min(1.0, float(items[key]["mae"]["p_value"]) * (len(ordered) - rank))
        running = max(running, value)
        items[key]["mae"]["holm_p_value"] = running


def main() -> None:
    require(P0.is_file() and json.loads(P0.read_text()).get("status") == "passed", "P0 checks have not passed")
    groups = {variant: [read_row(variant, seed) for seed in SEEDS]
              for variant in ("official_base", "feature_only", "permuted_q", "full")}
    data_hashes = {row["data_sha256"] for rows in groups.values() for row in rows}
    require(len(data_hashes) == 1, f"multiple data hashes: {sorted(data_hashes)}")
    for seed_index, seed in enumerate(SEEDS):
        hashes = {groups[variant][seed_index]["mask_sha256"] for variant in groups}
        require(len(hashes) == 1, f"unpaired mask for seed {seed}: {hashes}")
    require(all(groups["feature_only"][index]["parameter_count"] < groups["permuted_q"][index]["parameter_count"]
                for index in range(len(SEEDS))), "Feature-only must not match/exceed Full parameter count")

    summaries = {variant: metric_stats(rows) for variant, rows in groups.items()}
    full_mae = np.asarray(summaries["full"]["mae"]["values"])
    full_rmse = np.asarray(summaries["full"]["rmse"]["values"])
    comparisons = {}
    for variant in ("feature_only", "permuted_q"):
        comparisons[variant] = {
            "mae": paired(np.asarray(summaries[variant]["mae"]["values"]), full_mae),
            "rmse": paired(np.asarray(summaries[variant]["rmse"]["values"]), full_rmse),
        }
    holm(comparisons)
    result = {
        "status": "passed",
        "protocol": "METR-LA, Block-T 50%, 24-to-12, five paired seeds; unmasked future MAE/RMSE",
        "seeds": list(SEEDS), "n_result_rows": 20, "data_sha256": next(iter(data_hashes)),
        "groups": {key: {"metrics": summaries[key], "parameter_count": rows[0].get("parameter_count"),
                           "result_paths": [row["_path"] for row in rows]} for key, rows in groups.items()},
        "paired_variant_minus_full": comparisons,
        "permutation_invariants": json.loads(P0.read_text())["permutation"],
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
