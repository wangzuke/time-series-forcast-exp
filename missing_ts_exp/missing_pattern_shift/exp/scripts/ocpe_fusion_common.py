"""Versioned clean-data adapter and exactly paired, legacy-compatible masks."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import pathlib
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from protocol import DATASETS, DATA_ROOT, EXP_ROOT, _split_bounds, stable_seed, per_channel_point_mask
from posterior_s4_common import (seed_all, sha_file, digest, move_batch, memory_stats, runtime_metadata)
from ocpe_fusion_s4 import FusionS4, ConditionalEncoder, ModelConfig, VARIANTS, LABELS, make_optimizer

SUITE = "ocpe_fusion_uniform_r50_l96_h96_1120"
PROTOCOL = "ocpe_fusion_valid_weather_B_uniform_sep24_v2"


def source_fingerprints():
    paths = [EXP_ROOT / "models/ocpe_fusion_s4.py", EXP_ROOT / "scripts/ocpe_fusion_common.py",
             EXP_ROOT / "scripts/run_ocpe_fusion_s4.py", EXP_ROOT / "scripts/protocol.py",
             EXP_ROOT / "scripts/posterior_s4_common.py", EXP_ROOT / "models/posterior_s4.py",
             EXP_ROOT / "baselines/S4M/s4m/model/S4/s4/models/s4/s4.py"]
    return {str(p.relative_to(EXP_ROOT)): sha_file(p) for p in paths}


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


@lru_cache(maxsize=3)
def clean_dataset(dataset):
    path = DATA_ROOT / DATASETS[dataset]["file"]
    frame = pd.read_csv(path)
    names = [c for c in frame.columns if c != DATASETS[dataset]["date"]]
    raw = frame[names].to_numpy(dtype=np.float32)
    finite = np.isfinite(raw)
    sentinel = raw == -9999 if dataset == "Weather" else np.zeros_like(finite)
    # The newly audited, known issue is Weather's sentinel; unexpected NaNs fail visibly.
    if not finite.all():
        raise ValueError(f"Unexpected natural NaN/inf in {dataset}; inspect before extending the cleaning rule")
    valid = finite & ~sentinel
    train_end, val_end, test_end = _split_bounds(dataset, len(raw))
    train = np.where(valid[:train_end], raw[:train_end], np.nan).astype(np.float64)
    mean = np.nanmean(train, axis=0).astype(np.float32)
    std = (np.nanstd(train, axis=0) + 1e-8).astype(np.float32)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("A training variable has no valid values")
    raw_safe = np.where(valid, raw, 0).astype(np.float32)
    standardized = np.where(valid, (raw_safe - mean) / std, 0).astype(np.float32)
    quality = {"dataset": dataset, "raw_file": str(path), "data_sha256": sha_file(path),
               "rows": len(raw), "feature_names": names, "features": len(names),
               "split_bounds": [train_end, val_end, test_end], "validity_sha256": array_hash(valid),
               "mean": mean.tolist(), "std": std.tolist(), "mean_sha256": array_hash(mean), "std_sha256": array_hash(std),
               "cleaning": "Weather -9999 invalid; no rows removed; fit valid training values only; invalid targets excluded",
               "invalid_by_split": {}, "invalid_by_channel": (~valid).sum(0).tolist()}
    for label, left, right in [("train", 0, train_end), ("val", train_end, val_end), ("test", val_end, test_end)]:
        quality["invalid_by_split"][label] = {"total": int((~valid[left:right]).sum()),
                                             "by_channel": (~valid[left:right]).sum(0).tolist()}
    for array in (raw_safe, standardized, valid, mean, std):
        array.flags.writeable = False
    return raw_safe, standardized, valid, mean, std, (train_end, val_end, test_end), quality


class FusionDataset(Dataset):
    def __init__(self, dataset, split, family, seed, dynamic_epoch=False):
        if family not in ("P", "Bu") or split not in ("train", "val", "test"):
            raise ValueError((split, family))
        self.dataset, self.split, self.family = dataset, split, family
        self.seed, self.dynamic_epoch, self.epoch = int(seed), bool(dynamic_epoch), 0
        raw, x, valid, self.mean, self.std, bounds, self.quality = clean_dataset(dataset)
        train_end, val_end, test_end = bounds
        left, right = {"train": (0, train_end), "val": (train_end-96, val_end), "test": (val_end-96, test_end)}[split]
        self.x, self.raw, self.valid = x[left:right], raw[left:right], valid[left:right]
        self.n_features = x.shape[1]
        self.n_samples = right-left-191
        self.left = left

    def __len__(self):
        return self.n_samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def mask_for(self, index):
        epoch = self.epoch if self.dynamic_epoch else 0
        if self.family == "P":
            # Exactly the original P key, including otherwise irrelevant block_len=24.
            rng = np.random.default_rng(stable_seed(self.dataset, self.split, self.seed, epoch, index, "P", 24, "per_channel"))
            return per_channel_point_mask(96, self.n_features, .5, rng)
        if self.split == "test":
            rng = np.random.default_rng(stable_seed(self.dataset, self.split, self.seed, index, "B-uniform-v1"))
        else:
            rng = np.random.default_rng(stable_seed("B_uniform_sep24_v2", self.dataset, self.split, self.seed, epoch, index, "Bu"))
        out = np.ones((96, self.n_features), dtype=np.float32)
        for channel in range(self.n_features):
            a, b = np.sort(rng.choice(49, size=2, replace=False))
            s1, s2 = int(a), int(b)+24
            out[s1:s1+24, channel] = 0
            out[s2:s2+24, channel] = 0
        return out

    def effective_mask(self, index):
        return self.mask_for(index) * self.valid[index:index+96]

    def __getitem__(self, index):
        mask = self.effective_mask(index)
        target = slice(index+96, index+192)
        return {"x": torch.from_numpy(np.where(mask.astype(bool), self.x[index:index+96], 0).astype(np.float32)),
                "mask": torch.from_numpy(mask), "y": torch.from_numpy(self.x[target].copy()),
                "y_raw": torch.from_numpy(self.raw[target].copy()),
                "target_valid": torch.from_numpy(self.valid[target].astype(np.float32)),
                "index": torch.tensor(index, dtype=torch.long)}


def mask_fingerprint(ds, epoch=0, full=False, effective=False):
    saved = ds.epoch
    ds.set_epoch(epoch)
    h = hashlib.sha256()
    indices = range(len(ds)) if full else (0, min(17, len(ds)-1), len(ds)-1)
    for i in indices:
        h.update((ds.effective_mask(i) if effective else ds.mask_for(i)).tobytes())
    ds.set_epoch(saved)
    return h.hexdigest()


def mask_audit(ds, count=32):
    indices = np.linspace(0, len(ds)-1, min(count, len(ds)), dtype=int)
    masks = np.stack([ds.mask_for(int(i)) for i in indices])
    effective = np.stack([ds.effective_mask(int(i)) for i in indices])
    counts = (1-masks).sum(1)
    if not np.all(counts == 48):
        raise RuntimeError("Artificial mask must hide exactly 48 entries per channel")
    runs = []
    for mask in masks:
        for c in range(ds.n_features):
            edges = np.flatnonzero(np.diff(np.r_[1, mask[:, c], 1]))
            runs.extend((edges[1::2]-edges[::2]).tolist())
    if ds.family == "Bu" and set(runs) != {24}:
        raise RuntimeError("Bu must contain exactly separated B24 runs")
    return {"family": ds.family, "split": ds.split, "windows": len(indices),
            "artificial_missing_per_channel_min": float(counts.min()), "artificial_missing_per_channel_max": float(counts.max()),
            "missing_run_max": max(runs), "missing_probability_by_time": (1-masks).mean((0, 2)).tolist(),
            "effective_observed_min": float(effective.sum(1).min()),
            "effective_missing_rate": float(1-effective.mean())}
