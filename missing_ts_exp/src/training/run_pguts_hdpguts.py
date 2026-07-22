"""Training entrypoint for the 0721 P-GUTS / HD-PGUTS experiment line."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pathlib
import random
import subprocess
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ..models.pguts_hdpguts import (
    PGUTSConfig,
    PGUTSHDPGUTSForecaster,
    contiguous_coarse_assignment,
    parse_pooling_factors,
)


ROOT = pathlib.Path("/data/wangzuke/time-series-forecast-exp")
MASK_DIR = ROOT / "dataset" / "0721_missing_masks"
RESULTS_ROOT = ROOT / "missing_ts_exp" / "results" / "0721_cofill_pguts_forecasting"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def read_manifest_row(dataset: str, mask_type: str, rate: float) -> dict[str, str]:
    manifest = MASK_DIR / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest}")
    with manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            if (
                row["dataset"] == dataset
                and row["missing_type"] == mask_type
                and abs(float(row["target_missing_rate"]) - float(rate)) < 1e-9
            ):
                mask_path = pathlib.Path(row["mask_path"])
                actual_sha = sha256_file(mask_path)
                if actual_sha != row["mask_sha256"]:
                    raise ValueError(
                        f"mask sha256 mismatch for {mask_path}: {actual_sha} != {row['mask_sha256']}"
                    )
                return row
    raise ValueError(f"No manifest row for dataset={dataset} mask_type={mask_type} rate={rate}")


def load_split(dataset: str, horizon: int) -> dict[str, Any]:
    path = MASK_DIR / f"split_{dataset}_h{horizon}.json"
    with path.open() as f:
        return json.load(f)


def load_h5_matrix(path: pathlib.Path) -> tuple[np.ndarray, list[str]]:
    try:
        df = pd.read_hdf(path)
    except ImportError as exc:
        raise ImportError(
            "Reading METR-LA/PEMS-BAY .h5 files requires PyTables. "
            "Run this entrypoint in an environment with torch, pandas, and tables "
            "(for example set R0721_PYTHON_CMD='conda run -n <env> python'), "
            "or install pytables/tables into the active environment."
        ) from exc
    values = df.values.astype(np.float32)
    columns = [str(c) for c in df.columns]
    if values.ndim != 2:
        values = values.reshape(values.shape[0], -1)
    return values, columns


def read_distance_edges(distance_path: pathlib.Path) -> list[tuple[str, str, float]]:
    """Read DCRNN distance CSVs with or without a header row."""
    edges: list[tuple[str, str, float]] = []
    with distance_path.open(newline="") as f:
        reader = csv.reader(f)
        for row_idx, row in enumerate(reader):
            if not row:
                continue
            if row_idx == 0 and row[0].strip().lower() in {"from", "src", "source"}:
                continue
            if len(row) < 3:
                raise ValueError(f"Bad distance row in {distance_path}: {row}")
            src, dst, cost = row[0].strip(), row[1].strip(), float(row[2])
            edges.append((src, dst, cost))
    return edges


def load_adjacency(dataset: str, columns: list[str], graph_dir: pathlib.Path) -> np.ndarray:
    n_nodes = len(columns)
    if dataset == "Metr":
        distance_path = graph_dir / "distances_la_2012.csv"
        ids_path = graph_dir / "graph_sensor_ids.txt"
        if ids_path.exists():
            ids = ids_path.read_text().strip().split(",")
        else:
            ids = columns
    else:
        distance_path = graph_dir / "distances_bay_2017.csv"
        seen = set()
        ids = []
        for src, dst, _ in read_distance_edges(distance_path):
            for sensor_id in (src, dst):
                if sensor_id not in seen:
                    seen.add(sensor_id)
                    ids.append(sensor_id)

    if len(ids) != n_nodes:
        ids = columns
    id_to_idx = {str(sensor_id): idx for idx, sensor_id in enumerate(ids)}
    adj = np.eye(n_nodes, dtype=np.float32)
    costs: list[float] = []
    edges = read_distance_edges(distance_path)
    for src, dst, cost in edges:
        if src in id_to_idx and dst in id_to_idx and cost > 0:
            costs.append(cost)
    sigma = float(np.std(costs)) if costs else 1.0
    sigma = max(sigma, 1.0)
    for src, dst, cost in edges:
        if src in id_to_idx and dst in id_to_idx:
            weight = 1.0 if cost == 0 else math.exp(-cost / sigma)
            adj[id_to_idx[src], id_to_idx[dst]] = max(adj[id_to_idx[src], id_to_idx[dst]], weight)
    adj = np.maximum(adj, adj.T)
    adj = adj / np.maximum(adj.sum(axis=1, keepdims=True), 1e-6)
    return adj.astype(np.float32)


class TrafficMaskWindowDataset(Dataset):
    def __init__(
        self,
        values: np.ndarray,
        observed_mask: np.ndarray,
        starts: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        history: int,
        horizon: int,
    ):
        self.values = ((values - mean) / std).astype(np.float32)
        self.observed_mask = observed_mask.astype(np.float32)
        self.starts = starts.astype(np.int64)
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.history = int(history)
        self.horizon = int(horizon)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = int(self.starts[idx])
        hist = self.values[start : start + self.history]
        fut = self.values[start + self.history : start + self.history + self.horizon]
        hist_mask = self.observed_mask[start : start + self.history]
        x_hist = hist * hist_mask
        x_future = np.zeros((self.horizon, hist.shape[1]), dtype=np.float32)
        m_future = np.zeros_like(x_future)
        return {
            "x_all": torch.from_numpy(np.concatenate([x_hist, x_future], axis=0)[:, :, None]),
            "mask_all": torch.from_numpy(np.concatenate([hist_mask, m_future], axis=0)[:, :, None]),
            "y": torch.from_numpy(fut[:, :, None]),
            "history_mask": torch.from_numpy(hist_mask[:, :, None]),
        }


def split_starts(split: dict[str, Any], flag: str) -> np.ndarray:
    start = int(split[f"{flag}_start"])
    length = int(split[f"{flag}_len"])
    return np.arange(start, start + length, dtype=np.int64)


def make_loaders(args: argparse.Namespace, manifest_row: dict[str, str]):
    values, columns = load_h5_matrix(pathlib.Path(manifest_row["data_path"]))
    observed_mask = np.load(manifest_row["mask_path"])
    if observed_mask.ndim == 3:
        observed_mask = observed_mask[:, :, 0]
    split = load_split(args.dataset, args.T_out)
    if int(split["window"]) != args.T_in:
        raise ValueError(f"split window={split['window']} does not match T_in={args.T_in}")

    train_end_step = int(split["train_start"]) + int(split["train_len"]) + args.T_in + args.T_out - 1
    train_stats = values[:train_end_step]
    mean = train_stats.mean(axis=0, keepdims=True)
    std = train_stats.std(axis=0, keepdims=True) + 1e-6

    datasets = {
        flag: TrafficMaskWindowDataset(
            values,
            observed_mask,
            split_starts(split, flag),
            mean,
            std,
            args.T_in,
            args.T_out,
        )
        for flag in ("train", "val", "test")
    }
    loaders = {
        flag: DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=(flag == "train"),
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=(flag == "train"),
        )
        for flag, ds in datasets.items()
    }
    return loaders, values.shape[1], mean.squeeze(0), std.squeeze(0), columns, split


def limit_batches(loader, max_batches: int):
    for idx, batch in enumerate(loader):
        if max_batches >= 0 and idx >= max_batches:
            break
        yield batch


def original_scale(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return x * std.view(1, 1, -1, 1) + mean.view(1, 1, -1, 1)


def run_epoch(
    model: nn.Module,
    loader,
    optimizer,
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
    train: bool,
    grad_accum_steps: int,
    max_batches: int,
) -> dict[str, float]:
    model.train(train)
    loss_fn = nn.L1Loss()
    totals = {"loss": 0.0, "mae": 0.0, "rmse_num": 0.0, "mape_num": 0.0, "mre_num": 0.0, "mre_den": 0.0}
    n_values = 0
    n_batches = 0
    start = time.time()
    if train:
        optimizer.zero_grad(set_to_none=True)
    for batch_idx, batch in enumerate(limit_batches(loader, max_batches)):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.set_grad_enabled(train):
            pred = model(batch["x_all"], batch["mask_all"])
            loss = loss_fn(pred, batch["y"])
            if train:
                (loss / grad_accum_steps).backward()
                if (batch_idx + 1) % grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            pred_o = original_scale(pred, mean, std)
            y_o = original_scale(batch["y"], mean, std)
            err = pred_o - y_o
            abs_err = err.abs()
            totals["loss"] += float(loss.detach().item())
            totals["mae"] += float(abs_err.sum().item())
            totals["rmse_num"] += float((err ** 2).sum().item())
            totals["mape_num"] += float((abs_err / y_o.abs().clamp_min(1e-3)).sum().item())
            totals["mre_num"] += float(abs_err.sum().item())
            totals["mre_den"] += float(y_o.abs().sum().item())
            n_values += int(y_o.numel())
            n_batches += 1
    if train and n_batches % grad_accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return {
        "loss": totals["loss"] / max(1, n_batches),
        "MAE": totals["mae"] / max(1, n_values),
        "RMSE": math.sqrt(totals["rmse_num"] / max(1, n_values)),
        "MAPE": totals["mape_num"] / max(1, n_values),
        "MRE": totals["mre_num"] / max(totals["mre_den"], 1e-6),
        "time_sec": time.time() - start,
        "n_batches": float(n_batches),
    }


def evaluate_and_collect_predictions(
    model: nn.Module,
    loader,
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
    max_batches: int,
    collect_predictions: bool,
) -> tuple[dict[str, float], dict[str, np.ndarray] | None]:
    metrics = run_epoch(model, loader, None, device, mean, std, False, 1, max_batches)
    if not collect_predictions:
        return metrics, None
    ys: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    hist_masks: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in limit_batches(loader, max_batches):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            pred = model(batch["x_all"], batch["mask_all"])
            preds.append(original_scale(pred, mean, std).cpu().numpy())
            ys.append(original_scale(batch["y"], mean, std).cpu().numpy())
            hist_masks.append(batch["history_mask"].cpu().numpy())
    return metrics, {
        "y_true": np.concatenate(ys, axis=0),
        "y_pred": np.concatenate(preds, axis=0),
        "history_mask": np.concatenate(hist_masks, axis=0),
    }


def build_run_id(args: argparse.Namespace) -> str:
    pool = "-".join(str(v) for v in parse_pooling_factors(args.pooling_factors))
    rate = int(round(args.missing_rate * 100))
    model_tag = "pgutsf" if args.model == "pgutsf" else "hdpguts"
    return (
        f"pguts_{model_tag}_{args.dataset}_{args.mask_type}_r{rate}_h{args.T_out}_"
        f"pf{pool}_{args.variant}_s{args.seed}"
    )


def ensure_dirs(results_root: pathlib.Path) -> dict[str, pathlib.Path]:
    dirs = {
        "logs": results_root / "raw_logs" / "pguts_hdpguts",
        "checkpoints": results_root / "checkpoints" / "pguts_hdpguts",
        "csv": results_root / "csv",
        "notes": results_root / "notes",
        "predictions": results_root / "predictions",
        "metrics": results_root / "metrics" / "pguts_hdpguts",
        "diagnostics": results_root / "diagnostics" / "pguts_hdpguts",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["Metr", "PEMS"], required=True)
    parser.add_argument("--mask_type", choices=["point", "block_t", "block_st"], required=True)
    parser.add_argument("--missing_rate", type=float, required=True)
    parser.add_argument("--T_in", type=int, default=24)
    parser.add_argument("--T_out", type=int, default=24)
    parser.add_argument("--pooling_factors", default="3")
    parser.add_argument("--model", choices=["pgutsf", "hd_pguts"], default="pgutsf")
    parser.add_argument(
        "--variant",
        choices=["pguts", "no_graph_coarsening", "no_adaptive_fusion", "full"],
        default="pguts",
    )
    parser.add_argument("--graph_scale", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--allow_small_effective_batch", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--results_root", default=str(RESULTS_ROOT))
    parser.add_argument("--max_train_batches", type=int, default=-1)
    parser.add_argument("--max_eval_batches", type=int, default=-1)
    parser.add_argument("--save_predictions", action="store_true")
    args = parser.parse_args()

    effective_batch = args.batch_size * args.grad_accum_steps
    if effective_batch < 512 and not args.allow_small_effective_batch:
        raise ValueError(
            f"formal 0721 runs require effective batch >= 512; got "
            f"batch_size={args.batch_size} grad_accum_steps={args.grad_accum_steps}"
        )
    if args.model == "pgutsf" and args.variant != "pguts":
        raise ValueError("model=pgutsf must use variant=pguts")
    if args.model == "hd_pguts" and args.variant == "pguts":
        raise ValueError("model=hd_pguts must use an HD variant")

    set_seed(args.seed)
    results_root = pathlib.Path(args.results_root)
    dirs = ensure_dirs(results_root)
    run_id = build_run_id(args)
    manifest_row = read_manifest_row(args.dataset, args.mask_type, args.missing_rate)
    loaders, num_nodes, mean_np, std_np, columns, split = make_loaders(args, manifest_row)
    adjacency = torch.from_numpy(
        load_adjacency(args.dataset, columns, ROOT / "dataset" / "_archives" / "dcrnn_sensor_graph")
    )
    assignment = contiguous_coarse_assignment(num_nodes, args.graph_scale)

    device = torch.device(args.device)
    model = PGUTSHDPGUTSForecaster(
        PGUTSConfig(
            history=args.T_in,
            horizon=args.T_out,
            num_nodes=num_nodes,
            d_model=args.d_model,
            pooling_factors=parse_pooling_factors(args.pooling_factors),
            graph_scale=args.graph_scale,
            variant=args.variant,
            dropout=args.dropout,
        ),
        adjacency=adjacency,
        coarse_assignment=assignment,
    ).to(device)
    mean = torch.from_numpy(mean_np).float().to(device)
    std = torch.from_numpy(std_np).float().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(json.dumps({
        "run_id": run_id,
        "data_path": manifest_row["data_path"],
        "mask_path": manifest_row["mask_path"],
        "mask_sha256": manifest_row["mask_sha256"],
        "actual_missing_rate": manifest_row["actual_missing_rate"],
        "architecture_signature": model.architecture_signature(),
        "T_in": args.T_in,
        "T_out": args.T_out,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "effective_batch_size": effective_batch,
    }, ensure_ascii=True))

    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    train_start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, loaders["train"], optimizer, device, mean, std, True,
            args.grad_accum_steps, args.max_train_batches,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model, loaders["val"], None, device, mean, std, False, 1, args.max_eval_batches
            )
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            f"[{run_id}] ep={epoch}/{args.epochs} "
            f"train_MAE={train_metrics['MAE']:.4f} val_MAE={val_metrics['MAE']:.4f} "
            f"train_time={train_metrics['time_sec']:.1f}s"
        )
        if not math.isfinite(val_metrics["MAE"]):
            raise FloatingPointError(f"non-finite validation MAE at epoch {epoch}")
        if val_metrics["MAE"] < best_val - 1e-6:
            best_val = val_metrics["MAE"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"[{run_id}] early_stop epoch={epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics, prediction_pack = evaluate_and_collect_predictions(
        model,
        loaders["test"],
        device,
        mean,
        std,
        args.max_eval_batches,
        args.save_predictions,
    )
    train_time = time.time() - train_start

    checkpoint_path = dirs["checkpoints"] / f"{run_id}.pt"
    torch.save({"state_dict": model.state_dict(), "args": vars(args), "history": history}, checkpoint_path)

    prediction_path = ""
    if prediction_pack is not None:
        prediction_path = str(dirs["predictions"] / f"{run_id}.npz")
        metadata = {
            "run_id": run_id,
            "manifest_row": manifest_row,
            "args": vars(args),
            "split": split,
        }
        np.savez_compressed(
            prediction_path,
            y_true=prediction_pack["y_true"],
            y_pred=prediction_pack["y_pred"],
            target_mask=np.ones_like(prediction_pack["y_true"], dtype=np.float32),
            history_mask=prediction_pack["history_mask"],
            metadata_json=json.dumps(metadata, ensure_ascii=True),
        )

    weights_path = ""
    scale_weight_summary: dict[str, float] = {}
    if args.variant in {"full", "no_graph_coarsening"} and model.last_scale_weights is not None:
        weights = model.last_scale_weights.detach().cpu().numpy()
        weights_path = str(dirs["diagnostics"] / f"{run_id}_scale_weights.npy")
        np.save(weights_path, weights)
        for idx in range(weights.shape[1]):
            scale_weight_summary[f"scale_weight_{idx}_mean"] = float(weights[:, idx].mean())

    peak_mem = torch.cuda.max_memory_allocated(device) / 1024 ** 2 if device.type == "cuda" else 0.0
    row = {
        "run_id": run_id,
        "experiment_line": "0721_pguts_hdpguts",
        "model": args.model,
        "variant": args.variant,
        "source_code": "missing_ts_exp/src/training/run_pguts_hdpguts.py",
        "source_commit": source_commit(),
        "dataset": args.dataset,
        "num_nodes": num_nodes,
        "time_steps": manifest_row["n_timesteps"],
        "mask_type": args.mask_type,
        "target_missing_rate": f"{args.missing_rate:.2f}",
        "actual_missing_rate": manifest_row["actual_missing_rate"],
        "mask_sha256": manifest_row["mask_sha256"],
        "T_in": args.T_in,
        "T_out": args.T_out,
        "pooling_factors": args.pooling_factors,
        "graph_scale": args.graph_scale,
        "adaptive_fusion": args.variant in {"full", "no_graph_coarsening"},
        "architecture_signature": model.architecture_signature(),
        "seed": args.seed,
        "batch_size": effective_batch,
        "micro_batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "MAE": test_metrics["MAE"],
        "RMSE_or_MSE": test_metrics["RMSE"],
        "MAPE_or_MRE": test_metrics["MAPE"],
        "MRE": test_metrics["MRE"],
        "epoch_time_sec": np.mean([h["train"]["time_sec"] for h in history]) if history else 0.0,
        "train_time_sec": train_time,
        "gpu_peak_mb": peak_mem,
        "checkpoint_path": str(checkpoint_path),
        "log_path": str(dirs["logs"] / f"{run_id}.log"),
        "prediction_path": prediction_path,
        "scale_weights_path": weights_path,
        "notes": f"micro_batch={args.batch_size}; effective_batch={effective_batch}",
        **scale_weight_summary,
        "history": history,
        "test": test_metrics,
    }
    metrics_path = dirs["metrics"] / f"{run_id}.json"
    with metrics_path.open("w") as f:
        json.dump(row, f, indent=2, ensure_ascii=True)
    print(f"[{run_id}] wrote {metrics_path}")
    print(json.dumps({k: row[k] for k in ("run_id", "MAE", "RMSE_or_MSE", "MAPE_or_MRE")}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
