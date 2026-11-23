"""Isolated protocol helpers; never mutate the shared mask implementation."""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import random
import sys
import time

import numpy as np
import torch

from protocol import EXP_ROOT, ShiftForecastDataset, stable_seed

sys.path.insert(0, str(EXP_ROOT / "models"))
from posterior_s4 import ModelConfig, PosteriorS4, VARIANTS, make_optimizer, s4_module  # noqa: E402

SUITE = "posterior_s4_r50_b24_l96_h96_1119"


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def sha_file(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def source_fingerprints():
    files = [EXP_ROOT / "models/posterior_s4.py", EXP_ROOT / "scripts/posterior_s4_common.py",
             EXP_ROOT / "scripts/run_posterior_s4.py", EXP_ROOT / "scripts/protocol.py",
             EXP_ROOT / "baselines/S4M/s4m/model/S4/s4/models/s4/s4.py"]
    return {str(path.relative_to(EXP_ROOT)): sha_file(path) for path in files}


def digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class ExperimentDataset(ShiftForecastDataset):
    def __init__(self, dataset, split, family, seed, dynamic_epoch=False):
        self.uniform_blocks = family == "B-uniform"
        super().__init__(dataset, split, 96, 96, "B" if self.uniform_blocks else family,
                         0.5, seed, 24, dynamic_epoch, mask_balance="per_channel")

    def mask_for(self, index):
        if not self.uniform_blocks:
            return super().mask_for(index)
        rng = np.random.default_rng(stable_seed(self.dataset, self.split, self.seed, index, "B-uniform-v1"))
        out = np.ones((96, self.n_features), dtype=np.float32)
        # Weak compositions of 47 spare observed positions into 3 gaps:
        # choose two distinct separator slots uniformly from 49 slots.
        for channel in range(self.n_features):
            a, b = np.sort(rng.choice(49, size=2, replace=False))
            start1, start2 = int(a), int(b) + 24
            out[start1:start1 + 24, channel] = 0
            out[start2:start2 + 24, channel] = 0
        return out


def mask_fingerprint(ds, epoch=0):
    saved = ds.epoch
    ds.set_epoch(epoch)
    h = hashlib.sha256()
    for index in (0, min(17, len(ds) - 1), len(ds) - 1):
        h.update(ds.mask_for(index).tobytes())
    ds.set_epoch(saved)
    return h.hexdigest()


def mask_audit(ds, count=32):
    audit = ds.audit(count)
    indices = np.linspace(0, len(ds) - 1, min(count, len(ds)), dtype=int)
    masks = np.stack([ds.mask_for(int(i)) for i in indices])
    missing = 1 - masks
    counts = missing.sum(1)
    if not np.all(counts == 48):
        raise RuntimeError("Mask protocol violated: every channel must miss exactly 48 positions")
    last = np.where(masks.astype(bool), np.arange(96)[None, :, None], -1).max(1)
    audit.update(missing_probability_by_time=missing.mean((0, 2)).tolist(),
                 last_observed_position_mean=float(last.mean()),
                 last_observed_position_min=int(last.min()), last_observed_position_max=int(last.max()),
                 missing_per_window_channel_min=int(counts.min()), missing_per_window_channel_max=int(counts.max()),
                 adjacent_channel_joint_missing_fraction=float((missing[:, :, :-1] * missing[:, :, 1:]).mean()))
    return audit


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def memory_stats():
    return {"allocated_peak_mib": torch.cuda.max_memory_allocated() / 2**20,
            "reserved_peak_mib": torch.cuda.max_memory_reserved() / 2**20}


def runtime_metadata():
    module = s4_module()
    import structured_kernels
    return {"python": sys.executable, "torch": torch.__version__, "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0), "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "has_cuda_extension": module.has_cuda_extension,
            "extension_path": structured_kernels.__file__, "s4_path": module.__file__,
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "pid": os.getpid(), "timestamp": time.time()}
