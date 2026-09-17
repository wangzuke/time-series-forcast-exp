#!/usr/bin/env python3
"""Validate and summarize the 15 same-code 0917 protocol-closure runs."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats


ROOT = Path("/media/data1/wangzk/time-series-forcast-exp")
RESULT = ROOT / "missing_ts_exp/results/r0917_protocol_closed"
P0 = RESULT / "p0_audit.json"
OUT = RESULT / "verification_summary.json"
CSV = RESULT / "manifest_rows.csv"
SEEDS = (2024, 2025, 2026, 2027, 2028)
VARIANTS = ("full_observed", "feature_only", "full_permuted_q")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def tag(variant: str, seed: int) -> str:
    return f"{variant}_METRLA_blockt50_seed{seed}"


def checkpoint_exists(row: dict) -> bool:
    return bool(row.get("best_checkpoint")) and Path(row["best_checkpoint"]).is_file()


def finite(row: dict, section: str, field: str) -> float:
    value = float(row[section][field])
    require(math.isfinite(value), f"non-finite {section}.{field}: {row['_path']}")
    return value


def expected_model(variant: str) -> tuple[str, str | None, str]:
    if variant == "full_observed":
        return "mrmsgf", "full", "observed"
    if variant == "feature_only":
        return "official_base_feature_only", None, "not_applicable"
    if variant == "full_permuted_q":
        return "mrmsgf", "full", "permuted_pairwise"
    raise ValueError(variant)


def read_row(variant: str, seed: int) -> dict:
    path = RESULT / f"{tag(variant, seed)}.json"
    require(path.is_file(), f"missing result: {path}")
    row = json.loads(path.read_text())
    model, mr_variant, q_mode = expected_model(variant)
    require(row.get("protocol_id") == "r0917_protocol_closed", f"protocol mismatch: {path}")
    require(row.get("tag") == tag(variant, seed), f"tag mismatch: {path}")
    require(row.get("model") == model and row.get("mr_variant") == mr_variant and row.get("q_mode") == q_mode,
            f"model identity mismatch: {path}")
    require(row.get("seed") == seed and row.get("mask_seed") == seed, f"seed mismatch: {path}")
    require(row.get("epochs") == 100 and row.get("train_batches") == 300 and row.get("batch_size", 32) == 32,
            f"training budget mismatch: {path}")
    require(row.get("eval_batches") is None, f"formal task used smoke evaluation budget: {path}")
    require(row.get("optimizer") == {"name": "AdamW", "lr": 1e-3, "weight_decay": 0.01},
            f"optimizer mismatch: {path}")
    require(row.get("parameter_count", 0) > 0 and checkpoint_exists(row), f"checkpoint/parameter missing: {path}")
    require(row.get("best_epoch", -1) >= 0, f"best checkpoint epoch missing: {path}")
    require(row.get("window_metadata") == {
        "input_enabled": True,
        "input_fields": ["window_index", "split_id"],
        "consumed_for_permutation": variant == "full_permuted_q",
    }, f"window metadata mismatch: {path}")
    if variant == "feature_only":
        require(row.get("feature_only_no_dynamic_graph") is True, f"Feature-only architecture audit missing: {path}")
        require(row.get("zero_observation_physical_fallback") is False, f"Feature-only fallback flag mismatch: {path}")
    else:
        require(row.get("zero_observation_physical_fallback") is True and row.get("dispersion_numerical_protection") is True,
                f"Full safety manifest mismatch: {path}")
    if variant == "full_permuted_q":
        permutation = row.get("q_permutation") or {}
        require(permutation.get("transform") == "Q_perm=P Q P^T; diag(Q_perm)=diag(Q)",
                f"permutation transform mismatch: {path}")
        require(permutation.get("diagonal_restored") is True and permutation.get("off_diagonal_multiset_preserved") is True,
                f"permutation audit metadata mismatch: {path}")
    else:
        require(row.get("q_permutation") is None, f"unexpected permutation metadata: {path}")
    for section, fields in (("masked_test", ("test_mae", "test_mse")),
                            ("unmasked_test", ("test_mae_unmasked", "test_mse_unmasked"))):
        for field in fields:
            finite(row, section, field)
    row["_path"] = str(path)
    return row


def metric_stats(rows: list[dict]) -> dict[str, dict[str, object]]:
    mae = np.asarray([finite(row, "unmasked_test", "test_mae_unmasked") for row in rows])
    rmse = np.sqrt(np.asarray([finite(row, "unmasked_test", "test_mse_unmasked") for row in rows]))
    return {
        "mae": {"values": mae.tolist(), "mean": float(mae.mean()), "std": float(mae.std(ddof=1))},
        "rmse": {"values": rmse.tolist(), "mean": float(rmse.mean()), "std": float(rmse.std(ddof=1))},
    }


def paired(values: np.ndarray, reference: np.ndarray) -> dict[str, object]:
    delta = values - reference
    mean, std = float(delta.mean()), float(delta.std(ddof=1))
    if std == 0.0:
        ci, p_value = [mean, mean], (1.0 if mean == 0.0 else 0.0)
    else:
        half_width = float(stats.t.ppf(0.975, len(delta) - 1) * stats.sem(delta))
        ci = [mean - half_width, mean + half_width]
        p_value = float(stats.ttest_rel(values, reference).pvalue)
    return {"values": delta.tolist(), "mean": mean, "std": std, "ci95": ci, "p_value": p_value}


def holm(comparisons: dict[str, dict[str, object]]) -> None:
    ordered = sorted(comparisons, key=lambda key: float(comparisons[key]["mae"]["p_value"]))
    running = 0.0
    for rank, key in enumerate(ordered):
        adjusted = min(1.0, float(comparisons[key]["mae"]["p_value"]) * (len(ordered) - rank))
        running = max(running, adjusted)
        comparisons[key]["mae"]["holm_p_value"] = running


def full_config_signature(row: dict) -> dict:
    return {
        "protocol_id": row["protocol_id"],
        "model": row["model"],
        "mr_variant": row["mr_variant"],
        "seed": row["seed"],
        "mask_seed": row["mask_seed"],
        "epochs": row["epochs"],
        "train_batches": row["train_batches"],
        "eval_batches": row["eval_batches"],
        "patience": row["patience"],
        "batch_size": row["batch_size"],
        "graph_topk": row["graph_topk"],
        "reliability_beta": row["reliability_beta"],
        "artificial_missing_rate": row["artificial_missing_rate"],
        "model_hparams": row["model_hparams"],
        "optimizer": row["optimizer"],
        "region_assignment_sha256": row["region_assignment_sha256"],
        "data_sha256": row["data_sha256"],
        "mask_sha256": row["mask_sha256"],
        "parameter_count": row["parameter_count"],
        "window_metadata_input": {
            "input_enabled": row["window_metadata"]["input_enabled"],
            "input_fields": row["window_metadata"]["input_fields"],
        },
        "zero_observation_physical_fallback": row["zero_observation_physical_fallback"],
        "dispersion_numerical_protection": row["dispersion_numerical_protection"],
        "disable_overlap_reliability": row["disable_overlap_reliability"],
        "disable_readout_reliability": row["disable_readout_reliability"],
        "source_manifest": row["source_manifest"],
    }


def write_csv(rows: dict[str, list[dict]]) -> None:
    fields = ["variant", "seed", "tag", "model", "mr_variant", "q_mode", "parameter_count", "best_epoch",
              "best_val_mae", "test_mae_unmasked", "test_rmse_unmasked", "mask_sha256", "data_sha256",
              "source_head", "controlled_source_sha256", "best_checkpoint"]
    with CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for variant, group in rows.items():
            for row in group:
                writer.writerow({
                    "variant": variant, "seed": row["seed"], "tag": row["tag"], "model": row["model"],
                    "mr_variant": row["mr_variant"], "q_mode": row["q_mode"],
                    "parameter_count": row["parameter_count"], "best_epoch": row["best_epoch"],
                    "best_val_mae": row["best_val_mae"],
                    "test_mae_unmasked": row["unmasked_test"]["test_mae_unmasked"],
                    "test_rmse_unmasked": math.sqrt(row["unmasked_test"]["test_mse_unmasked"]),
                    "mask_sha256": row["mask_sha256"], "data_sha256": row["data_sha256"],
                    "source_head": row["source_manifest"]["head"],
                    "controlled_source_sha256": row["source_manifest"]["controlled_source_sha256"],
                    "best_checkpoint": row["best_checkpoint"],
                })


def main() -> None:
    require(P0.is_file() and json.loads(P0.read_text()).get("status") == "passed", "P0 audit did not pass")
    rows = {variant: [read_row(variant, seed) for seed in SEEDS] for variant in VARIANTS}
    source_heads = {row["source_manifest"]["head"] for group in rows.values() for row in group}
    source_hashes = {row["source_manifest"]["controlled_source_sha256"] for group in rows.values() for row in group}
    statuses = {row["source_manifest"]["status_porcelain"] for group in rows.values() for row in group}
    require(len(source_heads) == len(source_hashes) == len(statuses) == 1, "runs do not share one frozen source manifest")
    for index, seed in enumerate(SEEDS):
        paired = [rows[variant][index] for variant in VARIANTS]
        require(len({row["mask_sha256"] for row in paired}) == 1, f"unpaired masks for seed {seed}")
        require(len({row["data_sha256"] for row in paired}) == 1, f"unpaired data for seed {seed}")
        observed, permuted = rows["full_observed"][index], rows["full_permuted_q"][index]
        require(full_config_signature(observed) == full_config_signature(permuted),
                f"Full-observed and Permuted-Q configuration differs for seed {seed}")
    require(all(rows["full_observed"][index]["parameter_count"] == rows["full_permuted_q"][index]["parameter_count"]
                for index in range(len(SEEDS))), "Full and Permuted-Q parameter counts differ")

    summaries = {variant: metric_stats(group) for variant, group in rows.items()}
    full_mae, full_rmse = (np.asarray(summaries["full_observed"][metric]["values"])
                           for metric in ("mae", "rmse"))
    comparisons = {
        variant: {
            "mae": paired(np.asarray(summaries[variant]["mae"]["values"]), full_mae),
            "rmse": paired(np.asarray(summaries[variant]["rmse"]["values"]), full_rmse),
        }
        for variant in ("feature_only", "full_permuted_q")
    }
    holm(comparisons)
    write_csv(rows)
    result = {
        "status": "passed",
        "protocol": "METR-LA Block-T 50%, 24-to-12, same frozen source and metadata path, five paired seeds",
        "seeds": list(SEEDS), "n_result_rows": 15,
        "source_manifest": {"head": next(iter(source_heads)), "controlled_source_sha256": next(iter(source_hashes)),
                            "status_porcelain": next(iter(statuses))},
        "groups": {variant: {"metrics": summaries[variant], "parameter_count": group[0]["parameter_count"],
                              "result_paths": [row["_path"] for row in group]} for variant, group in rows.items()},
        "paired_variant_minus_full_observed": comparisons,
        "p0_audit": json.loads(P0.read_text()),
        "manifest_csv": str(CSV),
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
