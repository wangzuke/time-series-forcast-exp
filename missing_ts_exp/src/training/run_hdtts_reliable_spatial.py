"""Spatial runner for the HD-TTS-reliable idea model.

This entrypoint is intentionally separate from ``run_forecast.py`` because the
idea requires a physical/spatial graph. It supports the local METR-LA and AQI36
copies under ``dataset/`` and keeps official HD-TTS/GinAR repositories as
read-only references in ``external_repro/official/``.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import pickle
import random
import subprocess
import time
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ..data.hdtts_missing import HDTTSMaskParams, calibrate_block_t_p_fault, make_observed_mask
from ..data.missing import inject_missing
from ..models.hdtts_reliable import (
    ReliableHDTTSConfig,
    ReliableHDTTSForecaster,
    dense_greedy_assignment,
)


ROOT = pathlib.Path(__file__).resolve().parents[3]
DATA_ROOT = ROOT / "dataset"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def source_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def row_normalize_np(adj: np.ndarray) -> np.ndarray:
    adj = adj.astype(np.float32)
    denom = np.maximum(adj.sum(axis=-1, keepdims=True), 1e-6)
    return adj / denom


def gaussian_adj_from_latlng(latlng: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    lat = np.deg2rad(latlng[:, 0])
    lon = np.deg2rad(latlng[:, 1])
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    dist = 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    sigma = float(np.std(dist[dist > 0])) if np.any(dist > 0) else 1.0
    adj = np.exp(-(dist / max(sigma, 1e-6)) ** 2)
    adj[adj < threshold] = 0.0
    np.fill_diagonal(adj, 1.0)
    return row_normalize_np(adj)


def load_metr_la() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], str]:
    data_path = DATA_ROOT / "METR-LA" / "metr-la.h5"
    adj_path = DATA_ROOT / "METR-LA" / "adj_mx.pkl"
    df = pd.read_hdf(data_path)
    values = df.values.astype(np.float32)
    valid_mask = np.isfinite(values).astype(np.float32)
    values = np.nan_to_num(values, nan=0.0)
    with adj_path.open("rb") as f:
        sensor_ids, _, adj = pickle.load(f, encoding="latin1")
    return values, valid_mask, row_normalize_np(adj), [str(s) for s in sensor_ids], str(data_path)


def load_aqi36() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], str]:
    data_path = DATA_ROOT / "AQI" / "AQI36_PriSTI" / "pm25_ground.txt"
    latlng_path = DATA_ROOT / "AQI" / "AQI36_PriSTI" / "pm25_latlng.txt"
    df = pd.read_csv(data_path)
    station_cols = [c for c in df.columns if c != "datetime"]
    values = df[station_cols].to_numpy(dtype=np.float32)
    valid_mask = np.isfinite(values).astype(np.float32)
    values_df = pd.DataFrame(values).ffill().bfill().fillna(0.0)
    values = values_df.to_numpy(dtype=np.float32)
    latlng_df = pd.read_csv(latlng_path)
    latlng = latlng_df[[c for c in latlng_df.columns if c.lower() in {"latitude", "longitude", "lat", "lng", "lon"}]]
    if latlng.shape[1] < 2:
        latlng = latlng_df.iloc[:, -2:]
    adj = gaussian_adj_from_latlng(latlng.to_numpy(dtype=np.float32)[: len(station_cols)])
    return values, valid_mask, adj, station_cols, str(data_path)


def load_spatial_dataset(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], str]:
    key = name.lower()
    if key in {"metr-la", "metr_la", "metr"}:
        return load_metr_la()
    if key in {"aqi36", "aqi-36", "aqi"}:
        return load_aqi36()
    raise ValueError(f"Unsupported spatial dataset: {name}")


class SpatialWindowDataset(Dataset):
    def __init__(
        self,
        values: np.ndarray,
        valid_mask: np.ndarray,
        starts: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        seq_len: int,
        pred_len: int,
        missing_type: str,
        missing_rate: float,
        seed: int,
        injected_observed_mask: np.ndarray | None = None,
    ):
        self.values = ((values - mean) / std).astype(np.float32)
        self.valid_mask = valid_mask.astype(np.float32)
        self.starts = starts.astype(np.int64)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.missing_type = missing_type
        self.missing_rate = float(missing_rate)
        self.seed = int(seed)
        self.injected_observed_mask = None if injected_observed_mask is None else injected_observed_mask.astype(np.float32)

    def __len__(self) -> int:
        return int(len(self.starts))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = int(self.starts[idx])
        mid = start + self.seq_len
        end = mid + self.pred_len
        x = self.values[start:mid]
        y = self.values[mid:end]
        x_valid = self.valid_mask[start:mid]
        y_valid = self.valid_mask[mid:end]
        if self.injected_observed_mask is not None:
            injected = self.injected_observed_mask[start:mid]
        elif self.missing_type == "none" or self.missing_rate <= 0:
            injected = np.ones_like(x_valid, dtype=np.float32)
        else:
            sub_seed = (self.seed * 1315423911 + idx) & 0xFFFFFFFF
            missing_type = self.missing_type
            if missing_type == "point":
                missing_type = "random_point"
            elif missing_type == "block_t":
                missing_type = "continuous_segment"
            injected = inject_missing(
                x.shape,
                missing_type,
                self.missing_rate,
                seg_lengths=(6, 12, 24),
                seed=int(sub_seed),
            )
        mask = (x_valid * injected).astype(np.float32)
        return {
            "x": torch.from_numpy(x * mask).float(),
            "mask": torch.from_numpy(mask).float(),
            "y": torch.from_numpy(y).float(),
            "y_mask": torch.from_numpy(y_valid).float(),
        }


def split_starts(total_steps: int, seq_len: int, pred_len: int) -> dict[str, np.ndarray]:
    max_start = total_steps - seq_len - pred_len + 1
    train_end = int(max_start * 0.7)
    val_end = train_end + int(max_start * 0.1)
    return {
        "train": np.arange(0, train_end, dtype=np.int64),
        "val": np.arange(train_end, val_end, dtype=np.int64),
        "test": np.arange(val_end, max_start, dtype=np.int64),
    }


def load_observed_mask(path: str | None, expected_shape: tuple[int, int]) -> np.ndarray | None:
    if not path:
        return None
    mask = np.load(path).astype(np.float32)
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if mask.shape != expected_shape:
        raise ValueError(f"mask shape {mask.shape} does not match data shape {expected_shape}: {path}")
    return mask


def build_default_hdtts_mask(args: argparse.Namespace, values: np.ndarray) -> np.ndarray | None:
    if getattr(args, "mask_path", ""):
        return load_observed_mask(args.mask_path, values.shape)
    if args.missing_type == "point":
        params = HDTTSMaskParams(
            mode="point",
            p_fault=0.0,
            p_noise=float(args.missing_rate),
            min_seq=int(args.min_seq),
            max_seq=int(args.max_seq),
            seed=int(args.seed),
            target_missing_rate=float(args.missing_rate),
        )
        return make_observed_mask(values.shape[0], values.shape[1], params=params)
    if args.missing_type == "block_t":
        if args.calibrate_block_t:
            _, observed, _ = calibrate_block_t_p_fault(
                n_steps=values.shape[0],
                n_nodes=values.shape[1],
                target_missing_rate=float(args.missing_rate),
                seed=int(args.seed),
                p_noise=float(args.p_noise),
                min_seq=int(args.min_seq),
                max_seq=int(args.max_seq),
            )
            return observed
        params = HDTTSMaskParams(
            mode="block_t",
            p_fault=float(args.p_fault),
            p_noise=float(args.p_noise),
            min_seq=int(args.min_seq),
            max_seq=int(args.max_seq),
            seed=int(args.seed),
            target_missing_rate=float(args.missing_rate),
        )
        return make_observed_mask(values.shape[0], values.shape[1], params=params)
    return None


def masked_mse(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return ((pred - y).square() * mask).sum() / mask.sum().clamp_min(1.0)


def masked_mae(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return ((pred - y).abs() * mask).sum() / mask.sum().clamp_min(1.0)


def limit_batches(loader, max_batches: int):
    for idx, batch in enumerate(loader):
        if max_batches >= 0 and idx >= max_batches:
            break
        yield batch


def epoch_loop(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool,
    max_batches: int,
) -> dict[str, float]:
    model.train(train)
    total_mse = 0.0
    total_mae = 0.0
    total_n = 0
    started = time.time()
    for batch in limit_batches(loader, max_batches):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.set_grad_enabled(train):
            pred = model(batch["x"], batch["mask"])
            loss = masked_mse(pred, batch["y"], batch["y_mask"])
            if train:
                assert optimizer is not None
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        bsz = batch["y"].size(0)
        total_mse += float(loss.detach().item()) * bsz
        total_mae += float(masked_mae(pred.detach(), batch["y"], batch["y_mask"]).item()) * bsz
        total_n += bsz
    return {
        "mse": total_mse / max(1, total_n),
        "mae": total_mae / max(1, total_n),
        "n_samples": total_n,
        "time_sec": time.time() - started,
    }


def make_loaders(args: argparse.Namespace, values: np.ndarray, valid_mask: np.ndarray):
    starts = split_starts(len(values), args.seq_len, args.pred_len)
    injected_observed_mask = build_default_hdtts_mask(args, values)
    train_last = int(starts["train"][-1] + args.seq_len + args.pred_len) if len(starts["train"]) else args.seq_len
    train_values = values[:train_last]
    train_valid = valid_mask[:train_last]
    denom = train_valid.sum(axis=0).clip(min=1.0)
    mean = (train_values * train_valid).sum(axis=0) / denom
    var = (((train_values - mean) * train_valid) ** 2).sum(axis=0) / denom
    std = np.sqrt(var + 1e-6).astype(np.float32)
    mean = mean.astype(np.float32)
    datasets = {
        split: SpatialWindowDataset(
            values,
            valid_mask,
            split_starts_,
            mean,
            std,
            args.seq_len,
            args.pred_len,
            args.missing_type,
            args.missing_rate,
            args.seed,
            injected_observed_mask,
        )
        for split, split_starts_ in starts.items()
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=(split == "train"),
        )
        for split, dataset in datasets.items()
    }
    return loaders, mean, std, starts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="METR-LA", choices=["METR-LA", "AQI36"])
    parser.add_argument("--seq_len", type=int, default=24)
    parser.add_argument("--pred_len", type=int, default=12)
    parser.add_argument("--missing_type", default="variable_channel",
                        choices=["none", "random_point", "continuous_segment", "variable_channel", "mixed", "point", "block_t"])
    parser.add_argument("--missing_rate", type=float, default=0.5)
    parser.add_argument("--mask_path", default="", help="Path to a fixed observed mask .npy with 1=observed, 0=missing.")
    parser.add_argument("--p_noise", type=float, default=0.05)
    parser.add_argument("--p_fault", type=float, default=0.01)
    parser.add_argument("--min_seq", type=int, default=12)
    parser.add_argument("--max_seq", type=int, default=48)
    parser.add_argument("--calibrate_block_t", action="store_true")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", default=str(ROOT / "missing_ts_exp" / "results" / "r0902_hdtts_reliable"))
    parser.add_argument("--tag", default="")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--temporal_layers", type=int, default=2)
    parser.add_argument("--pooling_layers", type=int, default=2)
    parser.add_argument("--dilation", type=int, default=3)
    parser.add_argument("--max_cluster_size", type=int, default=8)
    parser.add_argument("--physical_graph_weight", type=float, default=1.0)
    parser.add_argument("--temporal_graph_weight", type=float, default=1.0)
    parser.add_argument("--adaptive_graph_weight", type=float, default=0.5)
    parser.add_argument("--max_train_batches", type=int, default=-1)
    parser.add_argument("--max_eval_batches", type=int, default=-1)
    args = parser.parse_args()

    set_seed(args.seed)
    values, valid_mask, adj, node_ids, data_path = load_spatial_dataset(args.dataset)
    loaders, mean, std, starts = make_loaders(args, values, valid_mask)

    device = torch.device(args.device)
    static_adj = torch.from_numpy(adj).float()
    assignment = dense_greedy_assignment(static_adj, args.max_cluster_size)
    cfg = ReliableHDTTSConfig(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        n_channels=values.shape[1],
        hidden_size=args.hidden,
        temporal_layers=args.temporal_layers,
        pooling_layers=args.pooling_layers,
        dilation=args.dilation,
        max_cluster_size=args.max_cluster_size,
        physical_graph_weight=args.physical_graph_weight,
        temporal_graph_weight=args.temporal_graph_weight,
        adaptive_graph_weight=args.adaptive_graph_weight,
    )
    model = ReliableHDTTSForecaster(cfg, static_adj=static_adj, assignment=assignment).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    bad = 0
    history = []
    for epoch in range(args.epochs):
        train_metrics = epoch_loop(model, loaders["train"], optimizer, device, True, args.max_train_batches)
        with torch.no_grad():
            val_metrics = epoch_loop(model, loaders["val"], None, device, False, args.max_eval_batches)
        history.append({"epoch": epoch + 1, "train": train_metrics, "val": val_metrics})
        print(
            f"[ep{epoch + 1}/{args.epochs}] "
            f"train_mse={train_metrics['mse']:.4f} train_mae={train_metrics['mae']:.4f} "
            f"val_mse={val_metrics['mse']:.4f} val_mae={val_metrics['mae']:.4f}"
        )
        if val_metrics["mse"] < best_val - 1e-8:
            best_val = val_metrics["mse"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    with torch.no_grad():
        test_metrics = epoch_loop(model, loaders["test"], None, device, False, args.max_eval_batches)

    out = {
        "model": "hdtts_reliable",
        "source_commit": source_commit(),
        "official_repos": {
            "hdtts": "external_repro/official/hdtts",
            "ginar": "external_repro/official/ginar",
        },
        "config": vars(args),
        "model_config": asdict(cfg),
        "data_path": data_path,
        "n_nodes": int(values.shape[1]),
        "node_ids_head": node_ids[:10],
        "adjacency_shape": list(adj.shape),
        "assignment_shape": list(assignment.shape),
        "split_sizes": {k: int(len(v)) for k, v in starts.items()},
        "valid_value_rate": float(valid_mask.mean()),
        "mask_path": args.mask_path,
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "best_val_mse": float(best_val),
        "test": test_metrics,
        "history": history,
    }
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or (
        f"{args.dataset}_{args.missing_type}_{int(args.missing_rate * 100)}_"
        f"h{args.pred_len}_seed{args.seed}"
    )
    out_path = out_dir / f"{tag}.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print("saved", out_path)


if __name__ == "__main__":
    main()
