#!/usr/bin/env python3
"""Collect post-hoc MR-MSGF reliability diagnostics from a saved checkpoint."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from missing_ts_exp.src.training.run_mrmsgf_official_controlled import build_predictor, make_datamodule


ROOT = Path("/media/data1/wangzk/time-series-forcast-exp")


class Moment:
    def __init__(self) -> None:
        self.n = 0
        self.sum = 0.0
        self.sum_sq = 0.0

    def add(self, value: torch.Tensor) -> None:
        value = value.detach().float()
        self.n += value.numel()
        self.sum += float(value.sum().cpu())
        self.sum_sq += float(value.square().sum().cpu())

    def output(self) -> dict[str, float]:
        mean = self.sum / max(1, self.n)
        variance = max(0.0, self.sum_sq / max(1, self.n) - mean * mean)
        return {"n": self.n, "mean": mean, "std": math.sqrt(variance)}


class Correlation:
    def __init__(self) -> None:
        self.n = 0
        self.x = 0.0
        self.y = 0.0
        self.xy = 0.0
        self.x2 = 0.0
        self.y2 = 0.0

    def add(self, x: torch.Tensor, y: torch.Tensor) -> None:
        x, y = x.detach().float(), y.detach().float()
        self.n += x.numel()
        self.x += float(x.sum().cpu())
        self.y += float(y.sum().cpu())
        self.xy += float((x * y).sum().cpu())
        self.x2 += float(x.square().sum().cpu())
        self.y2 += float(y.square().sum().cpu())

    def output(self) -> dict[str, float]:
        n = max(1, self.n)
        cov = self.xy / n - (self.x / n) * (self.y / n)
        vx = max(0.0, self.x2 / n - (self.x / n) ** 2)
        vy = max(0.0, self.y2 / n - (self.y / n) ** 2)
        return {"n": self.n, "pearson": cov / math.sqrt(vx * vy) if vx > 0 and vy > 0 else float("nan")}


def update_bucket(buckets: dict[str, list[float]], node_rate: torch.Tensor,
                  prediction: torch.Tensor, target: torch.Tensor, target_mask: torch.Tensor) -> None:
    # Prediction/target are B,H,N,C; diagnostics are B,N.
    abs_error = (prediction - target).abs() * target_mask
    valid = target_mask.sum(dim=(1, 3)).clamp_min(1.0)  # B,N
    node_mae = abs_error.sum(dim=(1, 3)) / valid
    for label, select in (
        ("zero_eq_0", node_rate == 0),
        ("low_gt_0_lt_0.4", (node_rate > 0) & (node_rate < 0.4)),
        ("medium_0.4_to_0.8", (node_rate >= 0.4) & (node_rate < 0.8)),
        ("high_ge_0.8", node_rate >= 0.8),
    ):
        values = node_mae[select]
        if values.numel():
            buckets[label][0] += float(values.sum().cpu())
            buckets[label][1] += int(values.numel())


def update_edge_selection_buckets(buckets: dict[str, list[int]], q_ij: torch.Tensor,
                                  selected: torch.Tensor, off_diagonal: torch.Tensor) -> None:
    """Accumulate top-k selection rates by pairwise co-observation bin."""
    for label, lower, upper in (
        ("[0.0,0.1)", 0.0, 0.1),
        ("[0.1,0.3)", 0.1, 0.3),
        ("[0.3,0.6)", 0.3, 0.6),
        ("[0.6,1.0]", 0.6, 1.0 + 1e-6),
    ):
        in_bin = off_diagonal & (q_ij >= lower) & (q_ij < upper)
        values = selected[in_bin]
        if values.numel():
            buckets[label][0] += int(values.sum().cpu())
            buckets[label][1] += int(values.numel())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_json", required=True)
    parser.add_argument("--out_path", default="")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_batches", type=int, default=-1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    row_path = Path(args.result_json).resolve()
    row = json.loads(row_path.read_text())
    if row.get("model") != "mrmsgf":
        raise ValueError("Reliability diagnostics require an MR-MSGF result JSON.")
    run_args = SimpleNamespace(
        model="mrmsgf",
        mr_variant=row.get("mr_variant") or "full",
        graph_topk=8,
        reliability_beta=1.0,
        q_mode=row.get("q_mode") or "observed",
        disable_overlap_reliability=bool(row.get("disable_overlap_reliability", False)),
        disable_readout_reliability=bool(row.get("disable_readout_reliability", False)),
    )
    mask_path = Path(row["mask_path"])
    datamodule, torch_dataset, adjacency, original_mask, dataset, data, input_mask = make_datamodule(mask_path, args.batch_size)
    predictor = build_predictor(run_args, torch_dataset, adjacency)
    checkpoints = sorted((row_path.parent / row["tag"]).glob("*.ckpt"))
    if len(checkpoints) != 1:
        raise ValueError(f"Expected one checkpoint for {row['tag']}, found {len(checkpoints)}")
    checkpoint = torch.load(checkpoints[0], map_location="cpu")
    predictor.load_state_dict(checkpoint["state_dict"])
    device = torch.device(args.device)
    predictor.to(device).eval()

    # Match the controlled runner's unmasked-future evaluation while retaining
    # the artificial input_mask supplied to the model.
    torch_dataset.set_mask(original_mask)
    torch_dataset.set_data(dataset.numpy())
    torch_dataset.add_covariate("x", data, "t n f", add_to_input_map=True, preprocess=True)
    torch_dataset.add_scaler("x", torch_dataset.scalers["target"])
    torch_dataset.add_covariate("input_mask", input_mask, "t n f", add_to_input_map=True)

    moments = {key: Moment() for key in (
        "node_observation_rate", "edge_common_observation_rate", "edge_gate",
        "region_coverage", "coarse_gate", "regional_dispersion", "block_fraction", "scale_attention",
    )}
    edge_correlation = Correlation()
    gate_correlations = {
        "node_observation_rate": Correlation(),
        "physical_neighbor_co_observation": Correlation(),
        "region_coverage": Correlation(),
        "longest_missing_fraction": Correlation(),
        "regional_dispersion": Correlation(),
    }
    edge_selection_buckets = {
        label: [0, 0] for label in ("[0.0,0.1)", "[0.1,0.3)", "[0.3,0.6)", "[0.6,1.0]")
    }
    buckets = {key: [0.0, 0] for key in (
        "zero_eq_0", "low_gt_0_lt_0.4", "medium_0.4_to_0.8", "high_ge_0.8"
    )}
    n_batches = 0
    with torch.no_grad():
        for batch in datamodule.test_dataloader():
            if args.max_batches >= 0 and n_batches >= args.max_batches:
                break
            batch = batch.to(device)
            target, prediction, target_mask = predictor.predict_batch(batch, return_target=True)
            diagnostic = predictor.model.last_diagnostics
            for key, moment in moments.items():
                moment.add(diagnostic[key])
            n_nodes = diagnostic["edge_common_observation_rate"].size(-1)
            off_diagonal = (~torch.eye(n_nodes, dtype=torch.bool, device=device).unsqueeze(0)).expand(
                diagnostic["edge_common_observation_rate"].size(0), -1, -1
            )
            edge_correlation.add(diagnostic["edge_common_observation_rate"][off_diagonal], diagnostic["edge_gate"][off_diagonal])
            update_edge_selection_buckets(
                edge_selection_buckets,
                diagnostic["edge_common_observation_rate"],
                diagnostic["edge_selected"],
                off_diagonal,
            )
            coarse_gate = diagnostic["coarse_gate"]
            gate_correlations["node_observation_rate"].add(coarse_gate, diagnostic["node_observation_rate"])
            gate_correlations["physical_neighbor_co_observation"].add(coarse_gate, diagnostic["physical_neighbor_co_observation"])
            gate_correlations["region_coverage"].add(coarse_gate, diagnostic["region_coverage"])
            gate_correlations["longest_missing_fraction"].add(coarse_gate, diagnostic["block_fraction"])
            gate_correlations["regional_dispersion"].add(coarse_gate, diagnostic["regional_dispersion"])
            update_bucket(buckets, diagnostic["node_observation_rate"], prediction, target, target_mask.float())
            n_batches += 1

    output = {
        "result_json": str(row_path),
        "checkpoint": str(checkpoints[0]),
        "mask_sha256": row["mask_sha256"],
        "data_sha256": row["data_sha256"],
        "n_test_batches": n_batches,
        "diagnostic_moments": {key: value.output() for key, value in moments.items()},
        "edge_q_gate_correlation_off_diagonal": edge_correlation.output(),
        "edge_selection_rate_by_q_bin": {
            key: {"selected": selected, "candidates": total, "rate": selected / total if total else float("nan")}
            for key, (selected, total) in edge_selection_buckets.items()
        },
        "coarse_gate_correlations": {key: value.output() for key, value in gate_correlations.items()},
        "mae_by_node_observation_rate": {
            key: {"n_nodes_across_windows": count, "mae": total / count if count else float("nan")}
            for key, (total, count) in buckets.items()
        },
    }
    out_path = Path(args.out_path) if args.out_path else row_path.parent / f"{row['tag']}_diagnostics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
