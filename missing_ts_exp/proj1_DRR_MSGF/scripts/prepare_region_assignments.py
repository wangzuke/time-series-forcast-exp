"""Build METR-LA partitions from the controlled runner's training split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..src.models.hdtts_official_reliable import graph_region_assignment
from ..src.training.run_mrmsgf_official_controlled import PROJECT_ROOT, make_datamodule, sha256


def functional_affinity(values: np.ndarray, mask: np.ndarray, topk: int) -> tuple[np.ndarray, np.ndarray]:
    observed = (mask & np.isfinite(values)).astype(np.float64)
    raw = np.where(observed > 0, values, 0.0).astype(np.float64)
    n = observed.sum(axis=0).clip(min=1)
    mean = raw.sum(axis=0) / n
    centered = np.where(observed > 0, raw - mean, 0.0)
    scale = np.sqrt(np.square(centered).sum(axis=0) / n).clip(min=1e-12)
    x = centered / scale
    support = observed.T @ observed
    sums = x.T @ observed
    safe_support = support.clip(min=1)
    variance = np.maximum(np.square(x).T @ observed - np.square(sums) / safe_support, 0)
    denominator = np.sqrt(variance * variance.T)
    covariance = x.T @ x - sums * sums.T / safe_support
    correlation = np.divide(covariance, denominator, out=np.zeros_like(covariance),
                            where=(support >= 3) & (denominator > 1e-12))
    correlation = np.clip(correlation, 0, 1)
    np.fill_diagonal(correlation, 0)
    neighbors = np.argsort(-correlation, axis=1, kind="stable")[:, :topk]
    sparse = np.zeros_like(correlation)
    np.put_along_axis(sparse, neighbors, np.take_along_axis(correlation, neighbors, axis=1), axis=1)
    return np.maximum(sparse, sparse.T).astype(np.float32), support.astype(np.int32)


def prepare(config: dict, out: Path) -> None:
    seed = config["development_seed"]
    mask_path = PROJECT_ROOT / config["mask_template"].format(seed=seed)
    dm, windows, _, original_mask, dataset, _, _ = make_datamodule(mask_path, batch_size=32)
    train = windows.expand_indices(dm.splitter.train_idxs, unique=True)["window"].numpy()
    held_out = np.concatenate([
        windows.expand_indices(split, unique=True, merge=True).numpy()
        for split in (dm.splitter.val_idxs, dm.splitter.test_idxs)
    ])
    if np.intersect1d(train, held_out).size:
        raise RuntimeError("Functional partition history overlaps held-out timestamps.")
    values = dataset.numpy().reshape(len(original_mask), -1)[train]
    valid = original_mask.reshape(len(original_mask), -1)[train]
    affinity, counts = functional_affinity(values, valid, config["graph_topk"])
    functional = graph_region_assignment(torch.from_numpy(affinity), config["max_region_size"]).numpy()
    physical_path = PROJECT_ROOT / config["physical_assignment"]
    physical = np.load(physical_path)
    random = physical[np.random.default_rng(seed).permutation(len(physical))]
    out.mkdir(parents=True, exist_ok=True)
    resources = {
        "training_history_indices": train,
        "functional_affinity": affinity,
        "pairwise_valid_counts": counts,
        "functional_static": functional,
        "random_size_matched": random,
    }
    hashes = {}
    for name, value in resources.items():
        path = out / f"{name}.npy"
        np.save(path, value)
        hashes[name] = sha256(path)
    manifest = {
        "training_history_count": len(train),
        "training_history_range": [int(train[0]), int(train[-1])],
        "overlap_with_held_out": 0,
        "uses_original_valid_mask": True,
        "physical_assignment_sha256": sha256(physical_path),
        "prepare_script_sha256": sha256(Path(__file__)),
        "artifact_sha256": hashes,
        "region_sizes": {
            "physical_static": physical.sum(axis=0).astype(int).tolist(),
            "functional_static": functional.sum(axis=0).astype(int).tolist(),
            "random_size_matched": random.sum(axis=0).astype(int).tolist(),
        },
    }
    (out / "region_resource_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(json.loads(args.config.read_text()), args.out)


if __name__ == "__main__":
    main()
