"""Structural checks for the revised MR-MSGF implementation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.utils import dense_to_sparse

from missing_ts_exp.src.models.hdtts_official_reliable import OfficialMRMSGFHDTTSModel
from missing_ts_exp.src.training.run_hdtts_reliable_spatial import load_spatial_dataset


ROOT = Path(__file__).resolve().parents[1]


def assert_connected_regions(adjacency: np.ndarray, assignment: np.ndarray) -> None:
    undirected = (adjacency + adjacency.T) > 0
    for region in range(assignment.shape[1]):
        members = np.flatnonzero(assignment[:, region] > 0.5)
        if not len(members):
            raise AssertionError(f"region {region} is empty")
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
        if len(seen) != len(member_set):
            raise AssertionError(f"region {region} is not physically connected")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="METR-LA", choices=["METR-LA", "AQI36"])
    parser.add_argument("--region_assignment_path", required=True)
    parser.add_argument("--seq_len", type=int, default=24)
    parser.add_argument("--max_region_size", type=int, default=8)
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()

    values, valid_mask, adjacency, _, _ = load_spatial_dataset(args.dataset)
    assignment = np.load(args.region_assignment_path).astype(np.float32)
    if assignment.shape[0] != adjacency.shape[0]:
        raise AssertionError("assignment node count does not match adjacency")
    if not np.allclose(assignment.sum(axis=1), 1.0):
        raise AssertionError("assignment is not one-hot by node")
    if int(assignment.sum(axis=0).max()) > args.max_region_size:
        raise AssertionError("assignment exceeds max_region_size")
    assert_connected_regions(adjacency, assignment)

    torch.manual_seed(7)
    device = torch.device("cpu")
    adj = torch.from_numpy(adjacency).float().to(device)
    edge_index, edge_weight = dense_to_sparse(adj)
    model = OfficialMRMSGFHDTTSModel(
        input_size=1,
        hidden_size=64,
        n_nodes=adj.size(0),
        horizon=12,
        rnn_layers=3,
        pooling_layers=3,
        mask_size=1,
        dilation=4,
        mp_method=["anisoconv", "anisoconv"],
        mp_kernel_size=1,
        recursive_lifting=True,
        fully_connected_readout=False,
        multi_step_scores=False,
        activation="elu",
        dropout=0.0,
        physical_adjacency=adj,
        region_assignment=torch.from_numpy(assignment),
        variant="full",
        graph_topk=8,
    ).to(device)
    model.eval()

    x0 = torch.from_numpy(values[: args.seq_len]).float().unsqueeze(0).unsqueeze(-1)
    m0 = torch.from_numpy(valid_mask[: args.seq_len]).float().unsqueeze(0).unsqueeze(-1)
    x1 = torch.from_numpy(values[1 : args.seq_len + 1]).float().unsqueeze(0).unsqueeze(-1)
    m1 = torch.from_numpy(valid_mask[1 : args.seq_len + 1]).float().unsqueeze(0).unsqueeze(-1)
    x0 = x0 * m0
    x1 = x1 * m1
    with torch.no_grad():
        single = model(x0, edge_index, edge_weight=edge_weight, input_mask=m0)[0]
        paired = model(torch.cat([x0, x1]), edge_index, edge_weight=edge_weight, input_mask=torch.cat([m0, m1]))[0][0:1]
        encoded = model.encoder_time(torch.cat([x0, x1]), mask=torch.cat([m0, m1]))
        observed, overlap, node_rate = model._observed_graph(encoded, torch.cat([m0, m1]).squeeze(-1))
        fine, _ = model._fine_graph(observed, overlap, node_rate)
        assignment_t = model.region_assignment
        coarse = torch.einsum("nk,bij,jl->bkl", assignment_t, fine, assignment_t)
        coarse = coarse / coarse.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    if not torch.allclose(single, paired, atol=args.atol, rtol=0.0):
        raise AssertionError("prediction changes when unrelated batch members are added")
    if not torch.isfinite(fine).all() or not torch.isfinite(coarse).all():
        raise AssertionError("Afine or Acoarse contains non-finite values")
    if not torch.allclose(fine.sum(dim=-1), torch.ones_like(fine.sum(dim=-1)), atol=args.atol):
        raise AssertionError("Afine is not row-normalized")
    if not torch.allclose(coarse.sum(dim=-1), torch.ones_like(coarse.sum(dim=-1)), atol=args.atol):
        raise AssertionError("Acoarse is not row-normalized")

    print(json.dumps({
        "dataset": args.dataset,
        "n_regions": int(assignment.shape[1]),
        "max_region_size": int(assignment.sum(axis=0).max()),
        "batch_invariance_max_abs_diff": float((single - paired).abs().max()),
        "afine_row_sum_max_error": float((fine.sum(dim=-1) - 1.0).abs().max()),
        "acoarse_row_sum_max_error": float((coarse.sum(dim=-1) - 1.0).abs().max()),
    }, indent=2))


if __name__ == "__main__":
    main()
