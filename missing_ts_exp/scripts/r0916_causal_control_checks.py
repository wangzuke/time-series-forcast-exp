#!/usr/bin/env python3
"""P0 structural and numerical checks for the 0916 causal controls."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path("/media/data1/wangzk/time-series-forcast-exp")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from missing_ts_exp.src.models.hdtts_official_reliable import permuted_pairwise_q
from missing_ts_exp.src.training.run_mrmsgf_official_controlled import build_predictor, make_datamodule


MASK = ROOT / "missing_ts_exp/resources/r0905_blockt/masks/METR-LA/block_t_r50_seed2024_observed.npy"
REGION = ROOT / "missing_ts_exp/resources/r0905_blockt/regions/METR-LA/physical_connected_max8_assignment.npy"
FORBIDDEN_FEATURE_MODULES = {"obs_query", "obs_key", "edge_gate", "fine_neigh", "fine_self",
                             "coarse_neigh", "coarse_self", "coarse_project", "coarse_gate"}


def args_for(model: str) -> argparse.Namespace:
    return argparse.Namespace(
        model=model, graph_topk=8, reliability_beta=1.0, mr_variant="full",
        q_mode="permuted_pairwise", mask_seed=2024,
        disable_overlap_reliability=False, disable_readout_reliability=False,
        region_assignment_path=str(REGION),
    )


def check_permutation() -> dict[str, bool]:
    values = torch.arange(49, dtype=torch.float32).reshape(7, 7)
    q = ((values + values.T) / 96.0).unsqueeze(0)
    window = torch.tensor([1234], dtype=torch.long)
    split = torch.tensor([1], dtype=torch.long)
    first = permuted_pairwise_q(q, window, split, 2024)
    second = permuted_pairwise_q(q, window, split, 2024)
    changed = permuted_pairwise_q(q, torch.tensor([1235]), split, 2024)
    checks = {
        "repeatable": bool(torch.equal(first, second)),
        "symmetric": bool(torch.equal(first, first.transpose(-1, -2))),
        "multiset_preserved": bool(torch.equal(torch.sort(q.flatten())[0], torch.sort(first.flatten())[0])),
        "range_preserved": bool(q.min() == first.min() and q.max() == first.max()),
        "different_window_changes_permutation": bool(not torch.equal(first, changed)),
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    return checks


def check_models() -> dict[str, object]:
    datamodule, dataset, adjacency, *_ = make_datamodule(MASK, batch_size=2, include_window_metadata=True)
    batch = next(iter(datamodule.train_dataloader()))
    full = build_predictor(args_for("mrmsgf"), dataset, adjacency).model
    feature = build_predictor(args_for("official_base_feature_only"), dataset, adjacency).model
    full.train()
    prediction = full(
        batch["x"], batch["edge_index"], input_mask=batch["input_mask"], u=batch["u"],
        window_index=batch["window_index"], split_id=batch["split_id"],
    )[0]
    prediction.mean().backward()
    feature_prediction = feature(
        batch["x"], batch["edge_index"], input_mask=batch["input_mask"], u=batch["u"],
    )[0]
    feature_prediction.mean().backward()

    zero_mask = batch["input_mask"].clone()
    zero_mask[:, :, 0, :] = False
    encoded = full.encoder_time(batch["x"], mask=zero_mask, u=batch["u"])
    a_obs, _, _, _, zero_nodes = full._observed_graph(
        encoded, full._mask_3d(zero_mask, batch["x"]), batch["window_index"], batch["split_id"]
    )
    zero_prediction = full(
        batch["x"], batch["edge_index"], input_mask=zero_mask, u=batch["u"],
        window_index=batch["window_index"], split_id=batch["split_id"],
    )[0]
    forbidden = {name.split(".")[0] for name, _ in feature.named_modules()} & FORBIDDEN_FEATURE_MODULES
    checks = {
        "full_forward_backward_finite": bool(torch.isfinite(prediction).all()),
        "feature_forward_backward_finite": bool(torch.isfinite(feature_prediction).all()),
        "zero_node_a_obs_is_zero": bool((a_obs[zero_nodes] == 0).all()),
        "zero_node_output_finite": bool(torch.isfinite(zero_prediction).all()),
        "feature_only_has_no_dynamic_modules": not forbidden,
        "feature_parameter_count": sum(parameter.numel() for parameter in feature.parameters()),
        "full_parameter_count": sum(parameter.numel() for parameter in full.parameters()),
    }
    if not all(value for key, value in checks.items() if isinstance(value, bool)):
        raise AssertionError(checks)
    if checks["feature_parameter_count"] >= checks["full_parameter_count"]:
        raise AssertionError(checks)
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "missing_ts_exp/results/r0916_causal_controls/p0_checks.json"))
    args = parser.parse_args()
    result = {"status": "passed", "permutation": check_permutation(), "models": check_models()}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
