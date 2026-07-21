#!/usr/bin/env python3
"""Build the five 0718 deliverable CSVs from metrics_summary.csv.

Reads missing_ts_exp/results/0718_block_hmbg/csv/metrics_summary.csv (produced
by r0718_collect_results.py) and writes, into the same csv/ directory:

  baseline_24to24.csv  -- BiTGraph + HD-TTS-AMP unified 24->24 rerun
  fusion_main.csv       -- Block-HMBGNet main-variant results
  ablation.csv          -- Block-HMBGNet ablation variants + deltas vs main
  efficiency.csv        -- n_params / batch size for all 3 models (timing columns
                           blank, see notes/implementation_notes.md)
  diagnostics.csv       -- header-only; gate/attention diagnostics were not
                           logged during Phase 4 runs (see notes/*.md)

Usage:
  python missing_ts_exp/scripts/r0718_build_csvs.py
"""

from __future__ import annotations

import csv
import pathlib

ROOT = pathlib.Path("/data/wangzuke/time-series-forecast-exp")
RESULTS = ROOT / "missing_ts_exp" / "results" / "0718_block_hmbg"
CSV_DIR = RESULTS / "csv"
SRC_CSV = CSV_DIR / "metrics_summary.csv"

COMMIT_BITGRAPH = "4f1d05bcc20bb3f5084bd1facc995478f84a40ed"
COMMIT_HDTTS = "46a8717db3802e1684e991ba0a7c0bfe43d22535"

# (n_nodes, n_timesteps) -- confirmed via external_repro/hdtts/calibrate_complete.py
# SHAPES dict and missing_ts_exp/results/0715_bitgraph_hdtts/csv/bitgraph_faithful.csv
DATASET_META = {
    ("BiTGraph", "Metr"): (207, 34272),
    ("BiTGraph", "PEMS"): (325, 52116),
    ("HD-TTS", "la"): (207, 34272),
    ("HD-TTS", "bay"): (325, 52128),
    ("Block-HMBGNet", "la"): (207, 34272),
    ("Block-HMBGNet", "bay"): (325, 52128),
}

# BiTGraph n_params confirmed from external_repro/BiTGraph/output_BiaTCGNet_*/best.pth
# checkpoints (architecture is rate/type-invariant per dataset).
BITGRAPH_NPARAMS = {"Metr": 114702, "PEMS": 172050}

ABLATION_VARIANTS = [
    "wo_gate",
    "wo_coarse",
    "wo_boundary",
    "graph_everywhere",
    "fine_only",
    "coarse_only",
]

PRIORITY_CONFIGS = [
    ("la", "block_t", "0.70"),
    ("la", "block_t", "0.80"),
    ("bay", "block_t", "0.70"),
    ("bay", "block_t", "0.80"),
    ("la", "block_st", "0.80"),
    ("bay", "block_st", "0.80"),
]


def load_rows() -> list[dict]:
    with SRC_CSV.open(newline="") as f:
        return list(csv.DictReader(f))


def is_real_experiment_row(row: dict) -> bool:
    """Exclude smoke/probe/calibration/preflight/infra log rows."""
    if row["status"] != "finished":
        return False
    stem = pathlib.Path(row["log_file"]).stem
    if stem.startswith("smoke_"):
        return False
    if row["model"] not in ("BiTGraph", "HD-TTS", "Block-HMBGNet"):
        return False
    # 5-epoch batch-size probes are not real accuracy runs.
    if row["epochs"] and row["epochs"] != "200":
        return False
    return True


# Wall-clock timing (train_time_per_epoch / total_train_time / sec_per_epoch)
# cannot be recovered for this round: BiTGraph's epoch-loss lines and HD-TTS's
# Lightning logs both lack per-epoch timestamps (checked directly -- BiTGraph
# prints only "epoch, loss: N <value>" with no clock time; HD-TTS's progress
# bar does not flush timestamped lines to a redirected log file), and the
# filesystem these logs live on does not expose st_birthtime (verified via
# os.stat -- hasattr is False), so log-file birth->mtime deltas (the 0715
# convention) are also unavailable here. This is a real, disclosed gap, not
# an oversight -- all timing columns below are left blank.


def hdtts_n_params(row: dict) -> str:
    """r0718_collect_results.py's PARAMS_RE captures only the numeric part of
    PyTorch Lightning's model-summary line (e.g. "588 K  Trainable params" ->
    row["n_params"]=="588"), silently dropping the K-thousands unit. All 0718
    HD-TTS/Block-HMBGNet runs report params in K (never M), so the true count
    is n_params*1000, rounded to the nearest thousand by Lightning itself
    (not an exact count)."""
    if not row["n_params"]:
        return ""
    return f"{int(float(row['n_params']) * 1000)}"


def actual_missing_rate(row: dict) -> str:
    """BiTGraph temporal_block prints an actual mask rate (target != realized
    due to block-length rounding); BiTGraph random_point constructs an exact
    zero count (get_0_1_array), so actual == target exactly; HD-TTS block_t /
    block_st use a pre-calibrated p_fault whose realized rate is documented in
    external_repro/hdtts/config/dataset/mode/*.yaml comments (la-canonical);
    HD-TTS point mode is analytically exact (p_noise == realized rate)."""
    if row["actual_missing_rate"]:
        return row["actual_missing_rate"]
    return row["missing_rate"]


# HD-TTS block_t/block_st actual realized missing rate, transcribed VERBATIM
# from binary-search calibration logs:
#   raw_logs/calibrate_0718_missing.log       (point_30, block_t_30, block_st_30-70,
#                                               block_st_80 initial [WARN] attempt)
#   raw_logs/calibrate_0718_block_st_80.log   (block_st_80 grid-scan re-calibration,
#                                               supersedes the [WARN] attempt above)
# point_50/60/70/80 are analytically exact by construction (p_fault=0, p_noise=
# target, same mechanism verified for point_30: la=0.3000/bay=0.3001), so no
# separate calibration log line exists for them -- this is not a gap.
# block_t_50/60/70/80: external_repro/hdtts/config/dataset/mode/block_t_{50,60,
# 70,80}.yaml carry ONLY a bare `p_fault` value with NO la_actual/bay_actual
# comment (unlike block_t_30 and every block_st_* config, which do) -- no
# calibration log for these four configs (on either dataset) exists anywhere
# in raw_logs/. This is a genuine, confirmed documentation gap, disclosed
# explicitly in the report rather than assumed equal to target.
HDTTS_ACTUAL_RATE = {
    ("la", "point", "0.30"): "0.3000",
    ("bay", "point", "0.30"): "0.3001",
    ("la", "block_t", "0.30"): "0.2976",
    ("bay", "block_t", "0.30"): "0.2973",
    ("la", "block_st", "0.30"): "0.3010",
    ("bay", "block_st", "0.30"): "0.2963",
    ("la", "block_st", "0.50"): "0.4985",
    ("bay", "block_st", "0.50"): "0.4986",
    ("la", "block_st", "0.60"): "0.5953",
    ("bay", "block_st", "0.60"): "0.6038",
    ("la", "block_st", "0.70"): "0.7037",
    ("bay", "block_st", "0.70"): "0.6957",
    ("la", "block_st", "0.80"): "0.8022",
    ("bay", "block_st", "0.80"): "0.8033",
}

# point_50/60/70/80: exact by construction (p_noise == target rate directly).
for _rate in ("0.50", "0.60", "0.70", "0.80"):
    HDTTS_ACTUAL_RATE[("la", "point", _rate)] = f"{_rate}(exact_by_construction)"
    HDTTS_ACTUAL_RATE[("bay", "point", _rate)] = f"{_rate}(exact_by_construction)"


def hdtts_actual_rate(dataset: str, mode: str, rate: str) -> str:
    key = (dataset, mode, rate)
    val = HDTTS_ACTUAL_RATE.get(key)
    if val is not None:
        return val
    # block_t_50/60/70/80 (both la and bay): no calibration log entry exists
    # anywhere in raw_logs/ -- fall back to the configured p_fault's target
    # rate and flag it as unverified rather than asserting equality.
    return f"{rate}(unverified_no_calibration_log)"


def build_baseline_csv(rows: list[dict]) -> None:
    fieldnames = [
        "paper", "model", "dataset", "n_nodes", "n_timesteps", "missing_type",
        "target_missing_rate", "actual_missing_rate", "window", "horizon",
        "batch_size", "seed", "mae", "mse", "rmse", "mape", "epochs",
        "best_epoch", "train_time_per_epoch", "total_train_time",
        "peak_gpu_mem_mb", "n_params", "commit_hash", "config_path", "notes",
    ]
    out = []
    for row in rows:
        if not is_real_experiment_row(row):
            continue
        if row["model"] == "BiTGraph":
            n_nodes, n_timesteps = DATASET_META[("BiTGraph", row["dataset"])]
            out.append({
                "paper": "BiTGraph",
                "model": "BiaTCGNet",
                "dataset": row["dataset"],
                "n_nodes": n_nodes,
                "n_timesteps": n_timesteps,
                "missing_type": row["missing_mode"],
                "target_missing_rate": row["missing_rate"],
                "actual_missing_rate": actual_missing_rate(row),
                "window": "24",
                "horizon": row["horizon"],
                "batch_size": row["batch_size"],
                "seed": "unset/random",
                "mae": row["mae"],
                "mse": f"{float(row['rmse']) ** 2:.4f}" if row["rmse"] else "",
                "rmse": row["rmse"],
                "mape": row["mape"],
                "epochs": row["epochs"],
                "best_epoch": "",
                "train_time_per_epoch": "",
                "total_train_time": "",
                "peak_gpu_mem_mb": "",
                "n_params": BITGRAPH_NPARAMS.get(row["dataset"], ""),
                "commit_hash": COMMIT_BITGRAPH,
                "config_path": "external_repro/BiTGraph (CLI args, no yaml config file)",
                "notes": (
                    "seed not logged (args.seed=-1 -> np.random.randint); "
                    "train_time_per_epoch/total_train_time blank -- no per-epoch "
                    "timestamps in log and filesystem lacks st_birthtime, see "
                    "notes/implementation_notes.md; mse=rmse^2; peak_gpu_mem_mb "
                    "not sampled this round (no concurrent nvidia-smi poller, "
                    "see notes/batch_size_notes.md)"
                ),
            })
        elif row["model"] == "HD-TTS":
            n_nodes, n_timesteps = DATASET_META[("HD-TTS", row["dataset"])]
            last_epoch = row["last_epoch"]
            out.append({
                "paper": "HD-TTS",
                "model": "hd_tts_amp",
                "dataset": row["dataset"],
                "n_nodes": n_nodes,
                "n_timesteps": n_timesteps,
                "missing_type": row["missing_mode"],
                "target_missing_rate": row["missing_rate"],
                "actual_missing_rate": hdtts_actual_rate(
                    row["dataset"], row["missing_mode"], row["missing_rate"]
                ),
                "window": "24",
                "horizon": row["horizon"],
                "batch_size": row["batch_size"],
                "seed": row["seed"],
                "mae": row["test_mae"],
                "mse": row["test_mse"],
                "rmse": f"{float(row['test_mse']) ** 0.5:.4f}" if row["test_mse"] else "",
                "mape": row["test_mre"],
                "epochs": row["epochs"],
                "best_epoch": (
                    str(int(last_epoch) + 1 - 30) if last_epoch else ""
                ),  # last completed epoch minus patience=30, approximate
                "train_time_per_epoch": "",
                "total_train_time": "",
                "peak_gpu_mem_mb": "",
                "n_params": hdtts_n_params(row),
                "commit_hash": COMMIT_HDTTS,
                "config_path": f"external_repro/hdtts/config/dataset/mode/{row['missing_mode']}_{int(float(row['missing_rate'])*100)}.yaml",
                "notes": (
                    "mae/mse/mape are the masked (primary) test metrics "
                    "(test_mae/test_mse/test_mre), matching 0715's convention; "
                    "best_epoch is approximate (last_epoch+1-patience, patience=30, "
                    "early stopping -- all runs stopped before reaching epoch 200); "
                    "n_params rounded to nearest 1000 (PyTorch Lightning model "
                    "summary reports params in K units, not an exact count); "
                    "train_time_per_epoch/total_train_time blank -- no per-epoch "
                    "timestamps in log and filesystem lacks st_birthtime; "
                    "peak_gpu_mem_mb not sampled this round"
                ),
            })
    out.sort(key=lambda r: (r["paper"], r["dataset"], r["missing_type"], r["target_missing_rate"]))
    write_csv(CSV_DIR / "baseline_24to24.csv", fieldnames, out)
    print(f"baseline_24to24.csv rows={len(out)}")


def build_fusion_main_csv(rows: list[dict]) -> None:
    fieldnames = [
        "model_variant", "dataset", "missing_type", "target_missing_rate",
        "actual_missing_rate", "window", "horizon", "batch_size", "seed",
        "mae", "mse", "rmse", "mape", "best_epoch", "gate_entropy",
        "coarse_gate_mean", "fine_gate_mean", "boundary_graph_weight_mean",
        "peak_gpu_mem_mb", "total_train_time", "n_params", "notes",
    ]
    out = []
    for row in rows:
        if not is_real_experiment_row(row):
            continue
        if row["model"] != "Block-HMBGNet" or row["variant"] != "main":
            continue
        out.append({
            "model_variant": "Block-HMBGNet_main",
            "dataset": row["dataset"],
            "missing_type": row["missing_mode"],
            "target_missing_rate": row["missing_rate"],
            "actual_missing_rate": hdtts_actual_rate(
                row["dataset"], row["missing_mode"], row["missing_rate"]
            ),
            "window": "24",
            "horizon": row["horizon"],
            "batch_size": row["batch_size"],
            "seed": row["seed"],
            "mae": row["test_mae"],
            "mse": row["test_mse"],
            "rmse": f"{float(row['test_mse']) ** 0.5:.4f}" if row["test_mse"] else "",
            "mape": row["test_mre"],
            "best_epoch": "",
            "gate_entropy": "",
            "coarse_gate_mean": "",
            "fine_gate_mean": "",
            "boundary_graph_weight_mean": "",
            "peak_gpu_mem_mb": "",
            "total_train_time": "",
            "n_params": hdtts_n_params(row),
            "notes": (
                "mae/mse/mape are masked (primary) test metrics; gate/attention "
                "diagnostics columns are blank -- block_hmbg_model.py computes "
                "gate_entropy/coarse_gate_mean/boundary_graph_weight_mean per "
                "forward() call, but predictor.py's test_step() discards the "
                "returned diagnostics dict without logging it (see diagnostics.csv "
                "notes); n_params rounded to nearest 1000 (Lightning model summary "
                "reports params in K units); total_train_time blank -- no per-epoch "
                "timestamps in log and filesystem lacks st_birthtime; peak_gpu_mem_mb "
                "not sampled this round"
            ),
        })
    out.sort(key=lambda r: (r["dataset"], r["missing_type"], r["target_missing_rate"]))
    write_csv(CSV_DIR / "fusion_main.csv", fieldnames, out)
    print(f"fusion_main.csv rows={len(out)}")


def build_ablation_csv(rows: list[dict]) -> None:
    fieldnames = [
        "base_model", "ablation", "dataset", "missing_type",
        "target_missing_rate", "seed", "mae", "mse", "rmse", "mape",
        "delta_mae_vs_full", "delta_pct_vs_full", "notes",
    ]
    by_key = {}
    for row in rows:
        if not is_real_experiment_row(row) or row["model"] != "Block-HMBGNet":
            continue
        key = (row["dataset"], row["missing_mode"], row["missing_rate"], row["variant"])
        by_key[key] = row

    out = []
    for dataset, mode, rate in PRIORITY_CONFIGS:
        main_row = by_key.get((dataset, mode, rate, "main"))
        main_mae = float(main_row["test_mae"]) if main_row else None
        for variant in ABLATION_VARIANTS:
            row = by_key.get((dataset, mode, rate, variant))
            if row is None:
                continue
            mae = float(row["test_mae"])
            delta = mae - main_mae if main_mae is not None else None
            delta_pct = (delta / main_mae * 100) if delta is not None and main_mae else None
            out.append({
                "base_model": "Block-HMBGNet_main",
                "ablation": variant,
                "dataset": dataset,
                "missing_type": mode,
                "target_missing_rate": rate,
                "seed": row["seed"],
                "mae": row["test_mae"],
                "mse": row["test_mse"],
                "rmse": f"{float(row['test_mse']) ** 0.5:.4f}" if row["test_mse"] else "",
                "mape": row["test_mre"],
                "delta_mae_vs_full": f"{delta:.4f}" if delta is not None else "",
                "delta_pct_vs_full": f"{delta_pct:.2f}" if delta_pct is not None else "",
                "notes": (
                    "delta = ablation_mae - main_mae for the same dataset/mode/rate "
                    "(positive = worse than full model); main reference row is not "
                    "itself included in this table (see fusion_main.csv)"
                ),
            })
    out.sort(key=lambda r: (r["dataset"], r["missing_type"], r["target_missing_rate"], r["ablation"]))
    write_csv(CSV_DIR / "ablation.csv", fieldnames, out)
    print(f"ablation.csv rows={len(out)}")


def build_efficiency_csv(rows: list[dict]) -> None:
    fieldnames = [
        "model", "dataset", "missing_type", "target_missing_rate",
        "batch_size", "epochs", "sec_per_epoch", "samples_per_sec",
        "total_time", "peak_gpu_mem_mb", "n_params", "gpu_id", "co_located",
        "notes",
    ]
    out = []
    for row in rows:
        if not is_real_experiment_row(row):
            continue
        if row["model"] == "BiTGraph":
            model_label = "BiaTCGNet"
            n_params = BITGRAPH_NPARAMS.get(row["dataset"], "")
        elif row["model"] == "HD-TTS":
            model_label = "hd_tts_amp"
            n_params = hdtts_n_params(row)
        elif row["model"] == "Block-HMBGNet":
            model_label = f"Block-HMBGNet_{row['variant']}"
            n_params = hdtts_n_params(row)
        else:
            continue
        out.append({
            "model": model_label,
            "dataset": row["dataset"],
            "missing_type": row["missing_mode"],
            "target_missing_rate": row["missing_rate"],
            "batch_size": row["batch_size"],
            "epochs": row["epochs"],
            "sec_per_epoch": "",
            "samples_per_sec": "",
            "total_time": "",
            "peak_gpu_mem_mb": "",
            "n_params": n_params,
            "gpu_id": "",
            "co_located": "true",
            "notes": (
                "co_located=true for ALL rows this round -- Phase 2b/Phase 4 "
                "runners packed up to 7 concurrent jobs across GPUs 1-7 with no "
                "single-job-per-GPU isolation window; per 实验计划0718.md section "
                "10.3, efficiency figures should therefore only use co_located=false "
                "rows, and this round provides none; sec_per_epoch/total_time/"
                "samples_per_sec/peak_gpu_mem_mb are blank -- no per-epoch "
                "timestamps in any log, filesystem lacks st_birthtime, and no "
                "concurrent nvidia-smi poller or throughput counter was attached "
                "-- see notes/batch_size_notes.md for the batch-probe-only "
                "throughput numbers that ARE available (HD-TTS it/s at batch "
                "64-512, isolated single-job measurement); n_params for "
                "hd_tts_amp/Block-HMBGNet rows is rounded to nearest 1000 "
                "(Lightning model summary reports params in K units)"
            ),
        })
    out.sort(key=lambda r: (r["model"], r["dataset"], r["missing_type"], r["target_missing_rate"]))
    write_csv(CSV_DIR / "efficiency.csv", fieldnames, out)
    print(f"efficiency.csv rows={len(out)}")


def build_diagnostics_csv() -> None:
    fieldnames = [
        "model_variant", "dataset", "missing_type", "target_missing_rate",
        "position", "gate_entropy", "coarse_gate_mean", "fine_gate_mean",
        "boundary_graph_weight_mean", "notes",
    ]
    write_csv(CSV_DIR / "diagnostics.csv", fieldnames, [])
    notes_path = RESULTS / "notes" / "diagnostics_gap.md"
    notes_path.write_text(
        "# diagnostics.csv gap\n\n"
        "`diagnostics.csv` is header-only (0 data rows).\n\n"
        "`external_repro/hdtts/lib/nn/models/block_hmbg_model.py`'s `forward()` "
        "computes `gate_entropy`, `coarse_gate_mean`, and "
        "`boundary_graph_weight_mean` per batch and returns them in a "
        "`diagnostics` dict (lines 187-193). However, "
        "`external_repro/hdtts/lib/nn/predictors/predictor.py`'s `predict_batch()` "
        "(lines 57-85) only unpacks `(y_hat, x_hat, scores, attn_weights)` from the "
        "model's forward output -- a 4-tuple `(out, None, alpha, diagnostics)` -- "
        "meaning `alpha` (the readout attention weights) is captured as "
        "`scores`/`attn_weights` positionally but the `diagnostics` dict itself is "
        "silently dropped, and neither is logged anywhere in `test_step()` "
        "(lines 203-217, which only calls `self.test_metrics.update(...)` and "
        "`self.log_loss(...)`). No Phase 4 raw log therefore contains "
        "`gate_entropy`/`coarse_gate_mean`/`fine_gate_mean`/"
        "`boundary_graph_weight_mean` values.\n\n"
        "Recovering this would require adding logging calls to `test_step()` (or a "
        "standalone post-hoc forward pass over saved checkpoints) and rerunning -- "
        "out of scope for this round. The 0718实验报告.md report should disclose "
        "this as an explicit limitation: gate-behavior claims in 实验计划0718.md's "
        "acceptance criterion 9.2.4 (block-center vs block-boundary gate routing) "
        "cannot be verified quantitatively from this round's artifacts.\n",
        encoding="utf-8",
    )
    print("diagnostics.csv rows=0 (see notes/diagnostics_gap.md)")


def write_csv(path: pathlib.Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    rows = load_rows()
    build_baseline_csv(rows)
    build_fusion_main_csv(rows)
    build_ablation_csv(rows)
    build_efficiency_csv(rows)
    build_diagnostics_csv()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
