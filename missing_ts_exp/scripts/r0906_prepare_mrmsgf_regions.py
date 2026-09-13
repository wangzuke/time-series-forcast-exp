"""Create and validate fixed connected regional assignments for MR-MSGF."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from missing_ts_exp.src.data.hdtts_missing import sha256_file
from missing_ts_exp.src.models.hdtts_official_reliable import graph_region_assignment
from missing_ts_exp.src.training.run_hdtts_reliable_spatial import load_spatial_dataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESOURCE_DIR = ROOT / "resources" / "r0905_blockt"


def region_is_connected(adj: np.ndarray, members: np.ndarray) -> bool:
    if len(members) <= 1:
        return True
    undirected = (adj + adj.T) > 0
    seen = {int(members[0])}
    frontier = [int(members[0])]
    member_set = {int(item) for item in members}
    while frontier:
        node = frontier.pop()
        for neighbor in np.flatnonzero(undirected[node]):
            neighbor = int(neighbor)
            if neighbor in member_set and neighbor not in seen:
                seen.add(neighbor)
                frontier.append(neighbor)
    return len(seen) == len(member_set)


def build_dataset(resource_dir: Path, dataset: str, max_region_size: int) -> dict[str, object]:
    _, _, adjacency, node_ids, _ = load_spatial_dataset(dataset)
    assignment = graph_region_assignment(torch.from_numpy(adjacency), max_region_size).cpu().numpy()
    region_sizes = assignment.sum(axis=0).astype(int)
    connected = []
    for region in range(assignment.shape[1]):
        members = np.flatnonzero(assignment[:, region] > 0.5)
        connected.append(region_is_connected(adjacency, members))
    if not all(connected):
        raise RuntimeError(f"{dataset}: generated a disconnected regional assignment")
    if int(region_sizes.max()) > max_region_size:
        raise RuntimeError(f"{dataset}: region size {int(region_sizes.max())} exceeds {max_region_size}")

    path = resource_dir / "regions" / dataset / f"physical_connected_max{max_region_size}_assignment.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, assignment.astype(np.float32))
    return {
        "dataset": dataset,
        "assignment_path": str(path),
        "assignment_sha256": sha256_file(path),
        "n_nodes": int(assignment.shape[0]),
        "n_regions": int(assignment.shape[1]),
        "max_region_size": int(max_region_size),
        "min_region_size": int(region_sizes.min()),
        "actual_max_region_size": int(region_sizes.max()),
        "region_sizes": region_sizes.tolist(),
        "all_regions_connected": bool(all(connected)),
        "node_ids": node_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource_dir", default=str(DEFAULT_RESOURCE_DIR))
    parser.add_argument("--datasets", default="METR-LA,AQI36")
    parser.add_argument("--max_region_size", type=int, default=8)
    args = parser.parse_args()

    resource_dir = Path(args.resource_dir)
    rows = [
        build_dataset(resource_dir, name.strip(), args.max_region_size)
        for name in args.datasets.split(",")
        if name.strip()
    ]
    out_path = resource_dir / "region_manifest.json"
    out_path.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"wrote {out_path}")
    for row in rows:
        print(row["dataset"], row["n_regions"], row["min_region_size"], row["actual_max_region_size"])


if __name__ == "__main__":
    main()
