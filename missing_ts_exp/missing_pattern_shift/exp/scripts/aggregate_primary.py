#!/usr/bin/env python3
"""Aggregate saved per-mask JSON results into tables and P->B paired deltas."""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from protocol import EXP_ROOT


def bootstrap_ci(values: np.ndarray, seed: int = 20261102) -> tuple[float, float]:
    if len(values) <= 1:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(10_000, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(samples, [0.025, 0.975]).tolist())


def paired_wilcoxon(values: np.ndarray) -> dict[str, float | int]:
    """Return transparent signed-rank statistics for a paired P→B penalty.

    The two-sided p value is the conventional omnibus result.  The ``greater``
    p value tests the preregistered directional alternative that the P→B
    penalty is positive.  Exact zero differences are omitted as prescribed by
    the Wilcoxon test and their count is retained for interpretation.
    """
    values = np.asarray(values, dtype=float)
    nonzero = values[values != 0]
    result: dict[str, float | int] = {
        "wilcoxon_nonzero_pairs": int(len(nonzero)),
        "wilcoxon_zero_pairs": int(len(values) - len(nonzero)),
        "wilcoxon_statistic": float("nan"),
        "wilcoxon_p_two_sided": float("nan"),
        "wilcoxon_p_greater": float("nan"),
    }
    if not len(nonzero):
        # Identical paired risks are evidence for no observed shift, not a
        # numerical failure.  A p value of one is conventional here.
        result.update({"wilcoxon_statistic": 0.0, "wilcoxon_p_two_sided": 1.0, "wilcoxon_p_greater": 1.0})
        return result
    two_sided = wilcoxon(values, zero_method="wilcox", alternative="two-sided", method="auto")
    greater = wilcoxon(values, zero_method="wilcox", alternative="greater", method="auto")
    result.update({
        "wilcoxon_statistic": float(two_sided.statistic),
        "wilcoxon_p_two_sided": float(two_sided.pvalue),
        "wilcoxon_p_greater": float(greater.pvalue),
    })
    return result


def selected_jobs(manifest: dict, baselines: set[str] | None) -> list[dict]:
    return [job for job in manifest["jobs"] if baselines is None or job["baseline"] in baselines]


def expected_paths(jobs: list[dict]) -> tuple[set[pathlib.Path], set[pathlib.Path]]:
    tests: set[pathlib.Path] = set()
    training: set[pathlib.Path] = set()
    for job in jobs:
        directory = EXP_ROOT / "results" / "primary" / job["id"]
        training.add(directory / "training_complete.json")
        for test in job["evaluations"]:
            tests.add(directory / f"test_{test['test_family']}_b{test['block_len']}_m{test['test_mask_seed']}.json")
    return tests, training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=pathlib.Path,
                        default=EXP_ROOT / "configs" / "primary_jobs_2seeds.json")
    parser.add_argument(
        "--out-dir", type=pathlib.Path, default=EXP_ROOT / "summary",
        help="Directory for this manifest's tables and Markdown report.",
    )
    parser.add_argument("--baseline", action="append", dest="baselines",
                        help="Only aggregate this baseline; repeat for multiple baselines.")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    jobs = selected_jobs(manifest, set(args.baselines) if args.baselines else None)
    expected_test_paths, expected_training_paths = expected_paths(jobs)
    rows = []
    step_rows = []
    runtime_rows = []
    for path in (EXP_ROOT / "results" / "primary").glob("**/test_*.json"):
        if path not in expected_test_paths:
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if record.get("status") != "complete":
            continue
        test = record.get("test", {})
        rows.append({
            "baseline": record["baseline"], "dataset": record["dataset"], "horizon": record["pred_len"],
            "train_family": record["train_family"], "test_family": record["test_family"],
            "block_len": record["block_len"], "train_seed": record["train_seed"],
            "test_mask_seed": record["test_mask_seed"], "mae": test["mae_raw"], "mse": test["mse_raw"],
            "objective": test["objective"], "n_windows": test["n_windows"], "path": str(path),
        })
        # Added after the first live batch of experiments.  Older atomic
        # results deliberately remain valid and simply do not contribute to
        # this optional horizon-resolved table.
        mae_by_step = test.get("mae_raw_by_step", [])
        mse_by_step = test.get("mse_raw_by_step", [])
        for step, (mae, mse) in enumerate(zip(mae_by_step, mse_by_step), start=1):
            step_rows.append({
                "baseline": record["baseline"], "dataset": record["dataset"], "horizon": record["pred_len"],
                "train_family": record["train_family"], "test_family": record["test_family"],
                "block_len": record["block_len"], "train_seed": record["train_seed"],
                "test_mask_seed": record["test_mask_seed"], "step": step, "mae_raw": mae, "mse_raw": mse,
                "path": str(path),
            })
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "primary_per_mask.csv", index=False)
    # Registered-versus-completed coverage, stratified at the granularity used
    # for the main scientific claims.  This deliberately counts only durable
    # test JSONs, not training/progress metadata.
    planned_rows = []
    for job in jobs:
        for test in job["evaluations"]:
            planned_rows.append({
                "baseline": job["baseline"], "dataset": job["dataset"], "horizon": job["pred_len"],
                "planned_test_evaluations": 1,
            })
    coverage_group = ["baseline", "dataset", "horizon"]
    planned = pd.DataFrame(planned_rows).groupby(coverage_group, dropna=False).size().rename("planned_test_evaluations")
    completed = frame.groupby(coverage_group, dropna=False).size().rename("completed_test_evaluations") if not frame.empty else pd.Series(dtype=int, name="completed_test_evaluations")
    completion = pd.concat([planned, completed], axis=1).fillna(0).reset_index()
    completion["completed_test_evaluations"] = completion["completed_test_evaluations"].astype(int)
    completion["remaining_test_evaluations"] = completion["planned_test_evaluations"] - completion["completed_test_evaluations"]
    completion["completion_fraction"] = completion["completed_test_evaluations"] / completion["planned_test_evaluations"]
    completion.to_csv(out / "primary_completion.csv", index=False)
    step_columns = [
        "baseline", "dataset", "horizon", "train_family", "test_family", "block_len", "train_seed",
        "test_mask_seed", "step", "mae_raw", "mse_raw", "path",
    ]
    step_frame = pd.DataFrame(step_rows, columns=step_columns)
    step_frame.to_csv(out / "primary_per_step.csv", index=False)
    if step_frame.empty:
        pd.DataFrame(columns=[
            "baseline", "dataset", "horizon", "train_family", "test_family", "block_len", "step",
            "repetitions", "mae_raw_mean", "mae_raw_std", "mse_raw_mean", "mse_raw_std",
        ]).to_csv(out / "primary_per_step_summary.csv", index=False)
    else:
        step_group = ["baseline", "dataset", "horizon", "train_family", "test_family", "block_len", "step"]
        step_frame.groupby(step_group, dropna=False).agg(
            repetitions=("mae_raw", "count"), mae_raw_mean=("mae_raw", "mean"), mae_raw_std=("mae_raw", "std"),
            mse_raw_mean=("mse_raw", "mean"), mse_raw_std=("mse_raw", "std"),
        ).reset_index().to_csv(out / "primary_per_step_summary.csv", index=False)
    # Training and evaluation run times live in separate durable artifacts.
    # Read them independently so progress/protocol JSONs cannot enter this
    # descriptive table.
    for path in (EXP_ROOT / "results" / "primary").glob("**/*.json"):
        if path.name.startswith("test_") and path not in expected_test_paths:
            continue
        if path.name == "training_complete.json" and path not in expected_training_paths:
            continue
        if not (path.name.startswith("test_") or path.name == "training_complete.json"):
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if record.get("status") != "complete":
            continue
        runtime_rows.append({
            "run_kind": "training" if path.name == "training_complete.json" else "evaluation",
            "baseline": record["baseline"], "dataset": record["dataset"], "horizon": record["pred_len"],
            "train_family": record["train_family"], "test_family": record["test_family"],
            "block_len": record["block_len"], "elapsed_sec": record.get("elapsed_sec"),
            "peak_gpu_allocated_mib": record.get("peak_gpu_allocated_mib"),
            "peak_gpu_reserved_mib": record.get("peak_gpu_reserved_mib"), "path": str(path),
        })
    runtime = pd.DataFrame(runtime_rows, columns=[
        "run_kind", "baseline", "dataset", "horizon", "train_family", "test_family", "block_len",
        "elapsed_sec", "peak_gpu_allocated_mib", "peak_gpu_reserved_mib", "path",
    ])
    runtime.to_csv(out / "primary_runtime_per_run.csv", index=False)
    if runtime.empty:
        pd.DataFrame(columns=[
            "run_kind", "baseline", "dataset", "horizon", "train_family", "test_family", "block_len",
            "runs", "elapsed_sec_mean", "elapsed_sec_std", "peak_gpu_allocated_mib_mean",
            "peak_gpu_reserved_mib_mean",
        ]).to_csv(out / "primary_runtime_summary.csv", index=False)
    else:
        runtime_group = ["run_kind", "baseline", "dataset", "horizon", "train_family", "test_family", "block_len"]
        runtime.groupby(runtime_group, dropna=False).agg(
            runs=("elapsed_sec", "count"), elapsed_sec_mean=("elapsed_sec", "mean"), elapsed_sec_std=("elapsed_sec", "std"),
            peak_gpu_allocated_mib_mean=("peak_gpu_allocated_mib", "mean"),
            peak_gpu_reserved_mib_mean=("peak_gpu_reserved_mib", "mean"),
        ).reset_index().to_csv(out / "primary_runtime_summary.csv", index=False)
    if frame.empty:
        (out / "primary_report.md").write_text("# Primary result summary\n\nNo completed primary results yet.\n", encoding="utf-8")
        return
    group = ["baseline", "dataset", "horizon", "train_family", "test_family", "block_len"]
    summary = frame.groupby(group, dropna=False).agg(
        repetitions=("mae", "count"), mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        mse_mean=("mse", "mean"), mse_std=("mse", "std"),
    ).reset_index()
    summary.to_csv(out / "primary_condition_summary.csv", index=False)
    # Pair P->B with B->B for the same baseline/data/H/block/train/test seeds.
    keys = ["baseline", "dataset", "horizon", "block_len", "train_seed", "test_mask_seed"]
    pb = frame[(frame.train_family == "P") & (frame.test_family == "B")][keys + ["mae", "mse"]].rename(columns={"mae": "mae_pb", "mse": "mse_pb"})
    bb = frame[(frame.train_family == "B") & (frame.test_family == "B")][keys + ["mae", "mse"]].rename(columns={"mae": "mae_bb", "mse": "mse_bb"})
    delta = pb.merge(bb, on=keys, how="inner")
    if not delta.empty:
        delta["delta_mae_pb_minus_bb"] = delta.mae_pb - delta.mae_bb
        delta["delta_mse_pb_minus_bb"] = delta.mse_pb - delta.mse_bb
    delta.to_csv(out / "primary_p_to_b_delta.csv", index=False)

    # Preserve the full 2×2 interpretation, rather than reporting only the
    # primary P→B-versus-B→B contrast.  P→P has no meaningful test block
    # length, so it is paired on the common experimental keys *without*
    # ``block_len`` and inherits the length of the block-side condition.
    common_keys = ["baseline", "dataset", "horizon", "train_seed", "test_mask_seed"]
    pp = frame[(frame.train_family == "P") & (frame.test_family == "P")][common_keys + ["mae", "mse"]]
    pb_for_effect = frame[(frame.train_family == "P") & (frame.test_family == "B")][common_keys + ["block_len", "mae", "mse"]]
    bp_for_effect = frame[(frame.train_family == "B") & (frame.test_family == "P")][common_keys + ["block_len", "mae", "mse"]]

    def effect_pairs(effect: str, left: pd.DataFrame, right: pd.DataFrame, merge_keys: list[str]) -> pd.DataFrame:
        merged = left.merge(right, on=merge_keys, how="inner", suffixes=("_left", "_right"))
        if merged.empty:
            return pd.DataFrame(columns=[
                "baseline", "dataset", "horizon", "block_len", "train_seed", "test_mask_seed", "effect",
                "mae_left", "mse_left", "mae_right", "mse_right", "delta_mae", "delta_mse",
            ])
        merged["effect"] = effect
        merged["delta_mae"] = merged["mae_left"] - merged["mae_right"]
        merged["delta_mse"] = merged["mse_left"] - merged["mse_right"]
        return merged[[
            "baseline", "dataset", "horizon", "block_len", "train_seed", "test_mask_seed", "effect",
            "mae_left", "mse_left", "mae_right", "mse_right", "delta_mae", "delta_mse",
        ]]

    effect_parts = []
    if not delta.empty:
        primary = delta.rename(columns={
            "mae_pb": "mae_left", "mse_pb": "mse_left", "mae_bb": "mae_right", "mse_bb": "mse_right",
            "delta_mae_pb_minus_bb": "delta_mae", "delta_mse_pb_minus_bb": "delta_mse",
        }).copy()
        primary["effect"] = "p_to_b_penalty"
        effect_parts.append(primary[[
            "baseline", "dataset", "horizon", "block_len", "train_seed", "test_mask_seed", "effect",
            "mae_left", "mse_left", "mae_right", "mse_right", "delta_mae", "delta_mse",
        ]])
    effect_parts += [
        effect_pairs("block_intrinsic_difficulty", pb_for_effect, pp, common_keys),
        effect_pairs("reverse_b_to_p_penalty", bp_for_effect, pp, common_keys),
    ]
    effect_frame = pd.concat(effect_parts, ignore_index=True)
    effect_frame.to_csv(out / "primary_2x2_effect_pairs.csv", index=False)
    effect_summary_rows = []
    if not effect_frame.empty:
        for keys_value, part in effect_frame.groupby(["baseline", "dataset", "horizon", "block_len", "effect"]):
            mae_ci_low, mae_ci_high = bootstrap_ci(part["delta_mae"].to_numpy())
            mse_ci_low, mse_ci_high = bootstrap_ci(part["delta_mse"].to_numpy())
            baseline, dataset, horizon, block_len, effect = keys_value
            row = {
                "baseline": baseline, "dataset": dataset, "horizon": horizon, "block_len": block_len,
                "effect": effect, "paired_repetitions": len(part),
                "delta_mae_mean": part["delta_mae"].mean(), "delta_mae_ci_low": mae_ci_low,
                "delta_mae_ci_high": mae_ci_high, "delta_mse_mean": part["delta_mse"].mean(),
                "delta_mse_ci_low": mse_ci_low, "delta_mse_ci_high": mse_ci_high,
            }
            row.update(paired_wilcoxon(part["delta_mae"].to_numpy()))
            effect_summary_rows.append(row)
    effect_columns = [
        "baseline", "dataset", "horizon", "block_len", "effect", "paired_repetitions",
        "delta_mae_mean", "delta_mae_ci_low", "delta_mae_ci_high",
        "delta_mse_mean", "delta_mse_ci_low", "delta_mse_ci_high",
        "wilcoxon_nonzero_pairs", "wilcoxon_zero_pairs", "wilcoxon_statistic",
        "wilcoxon_p_two_sided", "wilcoxon_p_greater",
    ]
    effect_summary = pd.DataFrame(effect_summary_rows, columns=effect_columns)
    effect_summary.to_csv(out / "primary_2x2_effect_summary.csv", index=False)
    # A machine-readable counterpart to the Markdown table below.  Keeping
    # the individual pairs above is useful for later tests; this compact
    # table is the one suitable for plotting/reporting condition-level
    # effects without reimplementing the pairing and bootstrap logic.
    delta_summary_rows = []
    if not delta.empty:
        for keys_value, part in delta.groupby(["baseline", "dataset", "horizon", "block_len"]):
            mae_ci_low, mae_ci_high = bootstrap_ci(part["delta_mae_pb_minus_bb"].to_numpy())
            mse_ci_low, mse_ci_high = bootstrap_ci(part["delta_mse_pb_minus_bb"].to_numpy())
            baseline, dataset, horizon, block_len = keys_value
            row = {
                "baseline": baseline,
                "dataset": dataset,
                "horizon": horizon,
                "block_len": block_len,
                "paired_repetitions": len(part),
                "delta_mae_mean": part["delta_mae_pb_minus_bb"].mean(),
                "delta_mae_ci_low": mae_ci_low,
                "delta_mae_ci_high": mae_ci_high,
                "delta_mse_mean": part["delta_mse_pb_minus_bb"].mean(),
                "delta_mse_ci_low": mse_ci_low,
                "delta_mse_ci_high": mse_ci_high,
            }
            row.update(paired_wilcoxon(part["delta_mae_pb_minus_bb"].to_numpy()))
            delta_summary_rows.append(row)
    pd.DataFrame(delta_summary_rows, columns=[
        "baseline", "dataset", "horizon", "block_len", "paired_repetitions",
        "delta_mae_mean", "delta_mae_ci_low", "delta_mae_ci_high",
        "delta_mse_mean", "delta_mse_ci_low", "delta_mse_ci_high",
        "wilcoxon_nonzero_pairs", "wilcoxon_zero_pairs", "wilcoxon_statistic",
        "wilcoxon_p_two_sided", "wilcoxon_p_greater",
    ]).to_csv(out / "primary_p_to_b_delta_summary.csv", index=False)
    lines = ["# Primary 2×2 result summary", "", f"Completed test evaluations: {len(frame)}.", ""]
    if not delta.empty:
        lines += ["## Paired P→B penalty", "", "`p(greater)` is a one-sided paired Wilcoxon signed-rank test of the preregistered alternative ΔMAE > 0; the full two-sided and directional results are in `primary_p_to_b_delta_summary.csv`.", "", "| Baseline | Dataset | H | Block | n | ΔMAE mean | 95% bootstrap CI | p(greater) |", "|---|---|---:|---:|---:|---:|---:|---:|"]
        for keys_value, part in delta.groupby(["baseline", "dataset", "horizon", "block_len"]):
            ci_low, ci_high = bootstrap_ci(part["delta_mae_pb_minus_bb"].to_numpy())
            base, dataset, horizon, block = keys_value
            mean_delta = part["delta_mae_pb_minus_bb"].mean()
            stats = paired_wilcoxon(part["delta_mae_pb_minus_bb"].to_numpy())
            lines.append(f"| {base} | {dataset} | {horizon} | {block} | {len(part)} | {mean_delta:.6f} | [{ci_low:.6f}, {ci_high:.6f}] | {stats['wilcoxon_p_greater']:.6g} |")
    else:
        lines += ["No complete matched P→B / B→B pairs yet; deltas will appear as workers finish both conditions."]
    auxiliary = effect_summary[effect_summary.effect != "p_to_b_penalty"]
    if not auxiliary.empty:
        effect_label = {
            "block_intrinsic_difficulty": "block intrinsic: P→B − P→P",
            "reverse_b_to_p_penalty": "reverse shift: B→P − P→P",
        }
        lines += ["", "## Auxiliary 2×2 effects", "", "| Effect | Baseline | Dataset | H | Block | n | ΔMAE mean | 95% bootstrap CI | p(greater) |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
        for _, row in auxiliary.sort_values(["effect", "baseline", "dataset", "horizon", "block_len"]).iterrows():
            lines.append(
                f"| {effect_label[row['effect']]} | {row['baseline']} | {row['dataset']} | {row['horizon']} | {row['block_len']} | "
                f"{row['paired_repetitions']} | {row['delta_mae_mean']:.6f} | [{row['delta_mae_ci_low']:.6f}, {row['delta_mae_ci_high']:.6f}] | {row['wilcoxon_p_greater']:.6g} |"
            )
    (out / "primary_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
