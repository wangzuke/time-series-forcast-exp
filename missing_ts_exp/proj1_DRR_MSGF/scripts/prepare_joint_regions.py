"""Export hard joint partitions from the frozen P1 edgewise graph generator."""
from __future__ import annotations

import argparse
import functools
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ..src.data.joint_regions import membership_changes, partition, partition_sweep


def describe(labels: np.ndarray, static: np.ndarray) -> dict:
    sizes = [np.bincount(row) for row in labels]
    pooled = np.concatenate(sizes)
    counts = np.asarray([len(row) for row in sizes])
    changes = [membership_changes(row, static).mean() for row in labels]
    return dict(
        windows=len(labels), region_count_min=int(counts.min()), region_count_max=int(counts.max()),
        region_count_mean=float(counts.mean()), max_size=int(pooled.max()),
        size_histogram=np.bincount(pooled).tolist(), fraction_over_eight=float((pooled > 8).mean()),
        mean_changed_nodes=float(np.mean(changes)),
    )


def prepare(config_path: Path, out: Path) -> None:
    import torch
    from tsl.data.loader import StaticGraphLoader

    from .regional_checkpoint import load_checkpoint
    from ..src.training.run_mrmsgf_official_controlled import PROJECT_ROOT, sha256

    config = json.loads(config_path.read_text())
    caps = config["caps"]
    parent = PROJECT_ROOT / config["parent_result"]
    row, dm, predictor = load_checkpoint(parent, config["batch_size"], "cuda")
    if row["fusion_mode"] != "edgewise" or row["mask_seed"] != config["seed"]:
        raise ValueError("Expected the matched P1 Edgewise checkpoint.")
    model = predictor.model
    model.freeze_joint_graph()
    windows = dm.torch_dataset
    loaders = {
        name: StaticGraphLoader(getattr(dm, name + "set"), batch_size=config["batch_size"],
                                shuffle=False, drop_last=False, num_workers=0)
        for name in ("train", "val", "test")
    }
    starts = {
        name: windows.get_window_indices(getattr(dm.splitter, name + "_idxs"))[:, 0].numpy()
        for name in loaders
    }
    all_starts = np.concatenate(list(starts.values()))
    if len(all_starts) != len(np.unique(all_starts)):
        raise ValueError("Expected globally unique window starts across the three splits.")

    def fused(batch):
        inputs = batch.to("cuda").input
        graph = model._graph_inputs(inputs["x"], inputs["input_mask"], inputs["u"])[-2]
        ids = inputs["window_index"].reshape(graph.size(0), -1)[:, 0].cpu().numpy()
        return ids, graph

    out.mkdir(parents=True, exist_ok=True)
    reference = torch.zeros((windows.n_nodes, windows.n_nodes), dtype=torch.float64, device="cuda")
    seen = []
    with torch.inference_mode():
        for batch in loaders["train"]:
            ids, graph = fused(batch)
            reference += graph.sum(dim=0, dtype=torch.float64)
            seen.extend(ids.tolist())
    if not np.array_equal(seen, starts["train"]):
        raise RuntimeError("Training reference did not cover exactly the training starts.")
    reference = (reference / len(seen)).cpu().numpy()
    weights = np.triu((reference + reference.T) * 0.5, 1)
    threshold = float(np.quantile(weights[weights > 0], config["threshold_quantile"]))
    np.save(out / "reference_graph.npy", reference)
    static = partition_sweep(reference, caps, threshold)
    for cap, labels in static.items():
        np.save(out / f"static_cap{cap}.npy", labels)
    print(f"Training reference: {len(seen)} windows, merge threshold {threshold:.6g}", flush=True)

    dynamic = {cap: np.full((int(all_starts.max()) + 1, windows.n_nodes), -1, dtype=np.int16)
               for cap in caps}
    solve = functools.partial(partition_sweep, caps=caps, threshold=threshold)
    with ProcessPoolExecutor(max_workers=config["partition_workers"],
                             mp_context=multiprocessing.get_context("spawn")) as pool, torch.inference_mode():
        for name, loader in loaders.items():
            seen = []
            for i, batch in enumerate(loader):
                ids, graph = fused(batch)
                seen.extend(ids.tolist())
                matrices = graph.cpu().numpy()
                if name == "val" and i == 0:
                    np.savez(out / "graph_audit.npz", starts=ids, graph=matrices)
                for start, labels_by_cap in zip(ids, pool.map(solve, matrices, chunksize=4)):
                    for cap, labels in labels_by_cap.items():
                        dynamic[cap][start] = labels
                if (i + 1) % 200 == 0 or i + 1 == len(loader):
                    print(f"Dynamic {name} partitions: {i + 1}/{len(loader)} batches", flush=True)
            if not np.array_equal(seen, starts[name]):
                raise RuntimeError(f"Unexpected coverage for {name}.")
    resources = {}
    summary = {}
    for cap in caps:
        labels = dynamic[cap][all_starts]
        if labels.min() < 0 or max(np.bincount(row).max() for row in labels) > cap:
            raise RuntimeError(f"Invalid partition produced for cap {cap}.")
        np.save(out / f"dynamic_cap{cap}.npy", dynamic[cap])
        summary[cap] = dict(static=describe(static[cap][None], static[cap]),
                            validation_dynamic=describe(dynamic[cap][starts["val"]], static[cap]))
        for kind in ("static", "dynamic"):
            path = out / f"{kind}_cap{cap}.npy"
            resources[path.name] = sha256(path)
    result = dict(config_sha256=sha256(config_path), teacher_checkpoint=row["best_checkpoint"],
                  teacher_sha256=sha256(Path(row["best_checkpoint"])), mask_sha256=row["mask_sha256"],
                  data_sha256=row["data_sha256"], physical_assignment_sha256=row["region_assignment_sha256"],
                  model_source_sha256=sha256(PROJECT_ROOT / "src/models/hdtts_official_reliable.py"),
                  partition_source_sha256=sha256(PROJECT_ROOT / "src/data/joint_regions.py"),
                  prepare_source_sha256=sha256(Path(__file__)), audit_sha256=sha256(out / "graph_audit.npz"),
                  reference_sha256=sha256(out / "reference_graph.npy"), threshold=threshold,
                  split_window_counts={k: len(v) for k, v in starts.items()}, labels_sha256=resources,
                  validation_partitions=summary)
    temp = out / "manifest.tmp"
    temp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temp.replace(out / "manifest.json")
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.config.resolve(), args.out.resolve())


if __name__ == "__main__":
    main()
