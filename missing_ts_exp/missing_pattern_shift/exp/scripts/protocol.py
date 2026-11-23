"""Shared data and mask protocol for the missing-pattern-shift experiments.

This module deliberately contains no forecasting model.  The three baseline
architectures are imported from their authors' repositories by
``run_official_baseline.py``.  Keeping the split, scaling, masks and metrics
here guarantees that P->P, P->B, B->P and B->B differ only in the declared
observation process.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


EXP_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_ROOT = EXP_ROOT / "data"

DATASETS: dict[str, dict[str, Any]] = {
    "ETTh1": {"file": "ETTh1.csv", "split": "ett", "date": "date", "freq": "h"},
    "Weather": {"file": "weather.csv", "split": "ratio", "date": "date", "freq": "t"},
    "Electricity": {"file": "electricity.csv", "split": "ratio", "date": "date", "freq": "h"},
    "Traffic": {"file": "traffic.csv", "split": "ratio", "date": "date", "freq": "h"},
}


def stable_seed(*parts: object) -> int:
    """Derive a process-independent uint32 seed (unlike Python's hash())."""
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def atomic_json_dump(payload: dict[str, Any], path: pathlib.Path) -> None:
    """Never leave a half-written metric/configuration file after interruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = pathlib.Path(handle.name)
    os.replace(temporary, path)


def atomic_torch_save(payload: object, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = pathlib.Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def _split_bounds(dataset: str, n_rows: int) -> tuple[int, int, int]:
    if DATASETS[dataset]["split"] == "ett":
        # Standard ETT hourly split: 12 / 4 / 4 months, 30 days per month.
        return 12 * 30 * 24, 16 * 30 * 24, 20 * 30 * 24
    train_end = int(n_rows * 0.7)
    return train_end, train_end + int(n_rows * 0.1), n_rows


def _read_dataset(dataset: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    if dataset not in DATASETS:
        raise KeyError(f"Unknown dataset: {dataset}; choose from {sorted(DATASETS)}")
    meta = DATASETS[dataset]
    path = DATA_ROOT / meta["file"]
    if not path.is_file():
        raise FileNotFoundError(f"Dataset missing: {path}")
    frame = pd.read_csv(path)
    features = [column for column in frame.columns if column != meta["date"]]
    values = frame[features].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"{dataset} contains natural NaN/inf values; this controlled protocol requires complete source data")
    train_end, val_end, test_end = _split_bounds(dataset, len(values))
    mean = values[:train_end].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = (values[:train_end].std(axis=0, dtype=np.float64) + 1e-8).astype(np.float32)
    return values, mean, std, train_end, val_end, test_end


def point_mask(length: int, channels: int, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Uniform point missingness with exactly round(rate*L*C) missing entries."""
    count = int(round(rate * length * channels))
    mask = np.ones(length * channels, dtype=np.float32)
    if count:
        mask[rng.choice(mask.size, size=count, replace=False)] = 0.0
    return mask.reshape(length, channels)


def per_channel_point_mask(length: int, channels: int, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Point masks with the same integer missing count in every channel."""
    return np.stack([point_mask(length, 1, rate, rng)[:, 0] for _ in range(channels)], axis=1)


def block_mask(length: int, channels: int, rate: float, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """Per-channel contiguous blocks with exactly the same missing count as P.

    Full blocks have ``block_len`` entries.  At most the final block is
    shortened to meet the matched per-window count exactly; this is recorded
    in the audit and avoids conflating missing-rate shift with pattern shift.
    """
    if block_len <= 0:
        raise ValueError("block_len must be positive for B masks")
    target = int(round(rate * length * channels))
    mask = np.ones((length, channels), dtype=np.float32)
    missing = 0
    # Each loop consumes a non-overlapping observed run.  Random feasible-start
    # proposals avoid materialising a C×L candidate list: this matters for the
    # 321-variable Electricity audit while preserving the same uniform channel
    # and start proposal distribution used in the small-data setting.
    for _ in range(max(1000, target * 8 + 1)):
        if missing >= target:
            break
        remaining = target - missing
        wanted = min(block_len, remaining)
        candidate: tuple[int, int] | None = None
        for _ in range(256):
            channel = int(rng.integers(channels))
            start = int(rng.integers(length - wanted + 1))
            if mask[start : start + wanted, channel].all():
                candidate = (channel, start)
                break
        if candidate is None:
            # Fallback after the window becomes crowded: find feasible runs
            # without retaining all candidates in memory.
            for channel in rng.permutation(channels):
                observed = mask[:, channel]
                valid = np.flatnonzero(np.convolve(observed, np.ones(wanted, dtype=np.float32), mode="valid") == wanted)
                if len(valid):
                    candidate = (int(channel), int(valid[int(rng.integers(len(valid)))]))
                    break
        if candidate is None:
            # Existing blocks make a nominal block impossible.  Finish with a
            # one-point observed position, preserving the exact rate.
            wanted = 1
            observed = np.argwhere(mask == 1)
            if not len(observed):
                break
            start, channel = observed[int(rng.integers(len(observed)))]
            candidate = (int(channel), int(start))
        channel, start = candidate
        mask[start : start + wanted, channel] = 0.0
        missing += wanted
    if missing != target:
        raise RuntimeError(f"Could not construct block mask: {missing=} {target=}")
    return mask


def per_channel_block_mask(length: int, channels: int, rate: float, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """Per-channel block masks with exact counts and separated blocks.

    The older rejection sampler allowed two nominal blocks to touch, making
    them indistinguishable from one longer outage.  Here each channel gets an
    exact partition of its observed positions into edge/inter-block gaps, so
    (for L=96, rate=.5, block_len=24) every channel has exactly two distinct
    24-step missing runs and at least one observed step between them.
    """
    if length <= 0 or channels <= 0 or block_len <= 0:
        raise ValueError("length, channels, and block_len must be positive")
    target = int(round(rate * length))
    if target < 0 or target > length:
        raise ValueError(f"invalid per-channel missing count {target} for length {length}")
    if target == 0:
        return np.ones((length, channels), dtype=np.float32)
    sizes = [block_len] * (target // block_len)
    remainder = target % block_len
    if remainder:
        sizes.append(remainder)
    if not sizes:
        sizes = [target]
    gaps_available = length - target - (len(sizes) - 1)
    if gaps_available < 0:
        raise ValueError("cannot place non-touching blocks at the requested missing rate")
    out = np.ones((length, channels), dtype=np.float32)
    probabilities = np.full(len(sizes) + 1, 1.0 / (len(sizes) + 1), dtype=np.float64)
    for channel in range(channels):
        gaps = rng.multinomial(gaps_available, probabilities)
        cursor = int(gaps[0])
        for number, size in enumerate(sizes):
            out[cursor : cursor + size, channel] = 0.0
            cursor += size
            if number < len(sizes) - 1:
                cursor += 1 + int(gaps[number + 1])
    return out


class ShiftForecastDataset(Dataset):
    """Time-ordered forecasting windows with independently reproducible masks."""

    def __init__(
        self,
        dataset: str,
        split: str,
        seq_len: int,
        pred_len: int,
        missing_family: str,
        missing_rate: float,
        seed: int,
        block_len: int,
        dynamic_epoch: bool,
        mask_balance: str = "global",
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(split)
        if missing_family not in {"P", "B"}:
            raise ValueError(missing_family)
        self.dataset, self.split = dataset, split
        self.seq_len, self.pred_len = int(seq_len), int(pred_len)
        self.missing_family, self.missing_rate = missing_family, float(missing_rate)
        if mask_balance not in {"global", "per_channel"}:
            raise ValueError(f"Unknown mask balance: {mask_balance}")
        self.seed, self.block_len, self.dynamic_epoch = int(seed), int(block_len), bool(dynamic_epoch)
        self.mask_balance = mask_balance
        self.epoch = 0
        raw, self.mean, self.std, train_end, val_end, test_end = _read_dataset(dataset)
        standardized = (raw - self.mean) / self.std
        if split == "train":
            left, right = 0, train_end
        elif split == "val":
            left, right = train_end - self.seq_len, val_end
        else:
            left, right = val_end - self.seq_len, test_end
        self.x = standardized[left:right].astype(np.float32, copy=False)
        self.y_raw = raw[left:right].astype(np.float32, copy=False)
        self.n_features = self.x.shape[1]
        self.n_samples = len(self.x) - self.seq_len - self.pred_len + 1
        if self.n_samples <= 0:
            raise ValueError(f"Split too short for L={seq_len}, H={pred_len}: {dataset}/{split}")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def mask_for(self, index: int) -> np.ndarray:
        epoch_component = self.epoch if self.dynamic_epoch else 0
        rng = np.random.default_rng(stable_seed(
            self.dataset, self.split, self.seed, epoch_component, index,
            self.missing_family, self.block_len, self.mask_balance,
        ))
        if self.mask_balance == "per_channel":
            if self.missing_family == "P":
                return per_channel_point_mask(self.seq_len, self.n_features, self.missing_rate, rng)
            return per_channel_block_mask(self.seq_len, self.n_features, self.missing_rate, self.block_len, rng)
        if self.missing_family == "P":
            return point_mask(self.seq_len, self.n_features, self.missing_rate, rng)
        return block_mask(self.seq_len, self.n_features, self.missing_rate, self.block_len, rng)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        history = self.x[index : index + self.seq_len]
        target_start = index + self.seq_len
        target = self.x[target_start : target_start + self.pred_len]
        target_raw = self.y_raw[target_start : target_start + self.pred_len]
        mask = self.mask_for(index)
        return {
            "x": torch.from_numpy(history * mask),
            "mask": torch.from_numpy(mask),
            "y": torch.from_numpy(target),
            "y_raw": torch.from_numpy(target_raw),
            "index": torch.tensor(index, dtype=torch.long),
        }

    def audit(self, count: int = 256) -> dict[str, float | int | str]:
        indices = np.linspace(0, self.n_samples - 1, num=min(count, self.n_samples), dtype=int)
        masks = np.stack([self.mask_for(int(index)) for index in indices])
        lengths: list[int] = []
        runs_per_window: list[int] = []
        affected_variable_fraction: list[float] = []
        for matrix in masks:
            window_runs = 0
            for channel in range(matrix.shape[1]):
                padded = np.concatenate(([1], matrix[:, channel], [1]))
                edges = np.flatnonzero(np.diff(padded) != 0)
                channel_lengths = (edges[1::2] - edges[::2]).tolist()
                lengths.extend(channel_lengths)
                window_runs += len(channel_lengths)
            runs_per_window.append(window_runs)
            affected_variable_fraction.append(float((matrix == 0).any(axis=0).mean()))
        return {
            "split": self.split,
            "family": self.missing_family,
            "mask_balance": self.mask_balance,
            "nominal_rate": self.missing_rate,
            "actual_rate_mean": float(1.0 - masks.mean()),
            "actual_rate_min": float(1.0 - masks.reshape(len(masks), -1).mean(axis=1).max()),
            "actual_rate_max": float(1.0 - masks.reshape(len(masks), -1).mean(axis=1).min()),
            "block_len_requested": self.block_len if self.missing_family == "B" else 0,
            "missing_run_mean": float(np.mean(lengths)) if lengths else 0.0,
            "missing_run_max": int(max(lengths)) if lengths else 0,
            "missing_runs_per_window_mean": float(np.mean(runs_per_window)) if runs_per_window else 0.0,
            "missing_runs_per_window_min": int(min(runs_per_window)) if runs_per_window else 0,
            "missing_runs_per_window_max": int(max(runs_per_window)) if runs_per_window else 0,
            "affected_variable_fraction_mean": float(np.mean(affected_variable_fraction)) if affected_variable_fraction else 0.0,
            "affected_variable_fraction_min": float(np.min(affected_variable_fraction)) if affected_variable_fraction else 0.0,
            "affected_variable_fraction_max": float(np.max(affected_variable_fraction)) if affected_variable_fraction else 0.0,
            "missing_per_channel_mean": float((1.0 - masks.mean(axis=(0, 1))).mean()),
            "missing_per_channel_min": float((1.0 - masks.mean(axis=(0, 1))).min()),
            "missing_per_channel_max": float((1.0 - masks.mean(axis=(0, 1))).max()),
            "audited_windows": int(len(indices)),
        }


def metrics_sums(pred: torch.Tensor, target_raw: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> tuple[float, float, int]:
    """Raw-unit MAE/MSE numerator and number of evaluated target elements."""
    pred_raw = pred * std.view(1, 1, -1) + mean.view(1, 1, -1)
    error = pred_raw - target_raw
    return float(error.abs().sum().item()), float(error.square().sum().item()), int(error.numel())
