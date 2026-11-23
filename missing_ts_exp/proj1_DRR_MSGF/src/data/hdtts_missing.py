"""HD-TTS-style missing mask utilities for Point and Block-T experiments.

The implementation mirrors the Point/Block-T branch of
``external_repro/official/hdtts/lib/utils/mask.py::sample_mask`` without the
Block-ST graph propagation branch. Masks returned by this module use
``1 = observed`` and ``0 = artificially missing`` so they can be multiplied
with dataset validity masks in local runners.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class HDTTSMaskParams:
    mode: str
    p_fault: float
    p_noise: float
    min_seq: int
    max_seq: int
    seed: int
    target_missing_rate: float | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sample_hdtts_missing_mask(
    shape: tuple[int, int, int],
    *,
    p_fault: float,
    p_noise: float,
    min_seq: int,
    max_seq: int,
    seed: int,
    dim: int = 0,
) -> np.ndarray:
    """Sample HD-TTS Point/Block-T artificial missing mask.

    Args:
        shape: ``(n_steps, n_nodes, n_channels)``.
        p_fault: Probability of starting a temporal fault at each position.
        p_noise: Independent point missing probability.
        min_seq: Minimum temporal fault length.
        max_seq: Maximum temporal fault length.
        seed: Torch generator seed.
        dim: Temporal dimension. HD-TTS uses ``dim=0`` for full datasets.

    Returns:
        Boolean array with ``True = artificially missing``.
    """
    if len(shape) != 3:
        raise ValueError(f"HD-TTS masks expect (steps, nodes, channels), got {shape}")
    if not 0 <= p_fault <= 1:
        raise ValueError(f"p_fault must be in [0, 1], got {p_fault}")
    if not 0 <= p_noise <= 1:
        raise ValueError(f"p_noise must be in [0, 1], got {p_noise}")
    min_seq = int(min_seq)
    max_seq = int(max_seq)
    if min_seq < 1 or max_seq < min_seq:
        raise ValueError(f"invalid fault length range: [{min_seq}, {max_seq}]")

    rng = torch.Generator()
    rng.manual_seed(int(seed))
    mask = torch.rand(shape, generator=rng) < float(p_fault)
    mask_idx = torch.nonzero(mask)
    if len(mask_idx):
        lens = torch.randint(min_seq, max_seq + 1, (len(mask_idx),), generator=rng)
        unmask_idx = mask_idx.clone()
        unmask_idx[:, dim] += lens
        n_faults = len(mask_idx)
        mask_idx = torch.repeat_interleave(mask_idx, lens, dim=0)
        increments = torch.arange(max_seq)[None]
        increments = torch.tile(increments, (n_faults, 1))
        increments = increments[increments < lens[:, None]]
        mask_idx[:, dim] += increments
        mask_idx = mask_idx[mask_idx[:, dim] < shape[dim]]
        if len(mask_idx):
            mask[tuple(mask_idx.T)] = True

        unmask_idx = unmask_idx[unmask_idx[:, dim] < shape[dim]]
    else:
        unmask_idx = mask_idx

    noise = torch.rand(shape, generator=rng) < float(p_noise)
    mask = torch.logical_or(mask, noise)

    if len(unmask_idx):
        mask[tuple(unmask_idx.T)] = False
    return mask.numpy()


def make_observed_mask(
    n_steps: int,
    n_nodes: int,
    *,
    params: HDTTSMaskParams,
    n_channels: int = 1,
) -> np.ndarray:
    missing = sample_hdtts_missing_mask(
        (int(n_steps), int(n_nodes), int(n_channels)),
        p_fault=params.p_fault,
        p_noise=params.p_noise,
        min_seq=params.min_seq,
        max_seq=params.max_seq,
        seed=params.seed,
        dim=0,
    )
    observed = (~missing).astype(np.float32)
    if n_channels == 1:
        return observed[..., 0]
    return observed


def actual_missing_rate(observed_mask: np.ndarray) -> float:
    return float(1.0 - np.asarray(observed_mask, dtype=np.float32).mean())


def calibrate_block_t_p_fault(
    *,
    n_steps: int,
    n_nodes: int,
    target_missing_rate: float,
    seed: int,
    p_noise: float = 0.05,
    min_seq: int = 12,
    max_seq: int = 48,
    n_iter: int = 18,
) -> tuple[float, np.ndarray, float]:
    """Binary-search ``p_fault`` for an approximate target missing rate."""
    target = float(target_missing_rate)
    if target <= p_noise:
        params = HDTTSMaskParams("block_t", 0.0, target, min_seq, max_seq, seed, target)
        obs = make_observed_mask(n_steps, n_nodes, params=params)
        return 0.0, obs, actual_missing_rate(obs)

    lo, hi = 0.0, 0.01
    best_pf = hi
    best_obs = None
    best_rate = -1.0
    while hi < 1.0:
        params = HDTTSMaskParams("block_t", hi, p_noise, min_seq, max_seq, seed, target)
        obs = make_observed_mask(n_steps, n_nodes, params=params)
        rate = actual_missing_rate(obs)
        best_pf, best_obs, best_rate = hi, obs, rate
        if rate >= target:
            break
        hi = min(1.0, hi * 2.0)

    for _ in range(int(n_iter)):
        mid = (lo + hi) / 2.0
        params = HDTTSMaskParams("block_t", mid, p_noise, min_seq, max_seq, seed, target)
        obs = make_observed_mask(n_steps, n_nodes, params=params)
        rate = actual_missing_rate(obs)
        if abs(rate - target) < abs(best_rate - target):
            best_pf, best_obs, best_rate = mid, obs, rate
        if rate < target:
            lo = mid
        else:
            hi = mid

    assert best_obs is not None
    return float(best_pf), best_obs, float(best_rate)
