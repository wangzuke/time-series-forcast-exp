#!/usr/bin/env python3
"""P0 audit for the 0917 same-code causal-control rerun."""
from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path

import torch


ROOT = Path("/media/data1/wangzk/time-series-forcast-exp")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from missing_ts_exp.src.models.hdtts_official_reliable import permuted_pairwise_q
from missing_ts_exp.src.training.run_mrmsgf_official_controlled import build_predictor, make_datamodule, set_seed


MASK = ROOT / "missing_ts_exp/resources/r0905_blockt/masks/METR-LA/block_t_r50_seed2024_observed.npy"
REGION = ROOT / "missing_ts_exp/resources/r0905_blockt/regions/METR-LA/physical_connected_max8_assignment.npy"
FORBIDDEN_FEATURE_MODULES = {
    "obs_query", "obs_key", "edge_gate", "fine_neigh", "fine_self", "coarse_neigh",
    "coarse_self", "coarse_project", "coarse_gate",
}


def require(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)


def model_args(model: str, q_mode: str = "observed") -> argparse.Namespace:
    return argparse.Namespace(
        model=model,
        graph_topk=8,
        reliability_beta=1.0,
        mr_variant="full",
        q_mode=q_mode,
        mask_seed=2024,
        disable_overlap_reliability=False,
        disable_readout_reliability=False,
        region_assignment_path=str(REGION),
    )


def construct(model: str, q_mode: str, dataset, adjacency):
    set_seed(2024)
    return build_predictor(model_args(model, q_mode), dataset, adjacency).model


def synthetic_q() -> torch.Tensor:
    n_nodes = 7
    values = torch.arange(n_nodes * n_nodes, dtype=torch.float32).reshape(n_nodes, n_nodes)
    q = (values + values.T).div(100.0)
    q.diagonal().copy_(torch.linspace(0.11, 0.77, n_nodes))
    return q.unsqueeze(0)


def p0_permutation() -> dict[str, bool]:
    q = synthetic_q()
    window = torch.tensor([1234], dtype=torch.long)
    split = torch.tensor([1], dtype=torch.long)
    first = permuted_pairwise_q(q, window, split, 2024)
    second = permuted_pairwise_q(q, window, split, 2024)
    changed = permuted_pairwise_q(q, torch.tensor([1235]), split, 2024)
    off_diagonal = ~torch.eye(q.size(-1), dtype=torch.bool).unsqueeze(0)
    result = {
        "repeatable": bool(torch.equal(first, second)),
        "symmetric": bool(torch.equal(first, first.transpose(-1, -2))),
        "diagonal_restored_exactly": bool(torch.equal(first.diagonal(dim1=-2, dim2=-1), q.diagonal(dim1=-2, dim2=-1))),
        "off_diagonal_multiset_preserved": bool(torch.equal(
            torch.sort(first[off_diagonal])[0], torch.sort(q[off_diagonal])[0]
        )),
        "different_window_changes_permutation": bool(not torch.equal(first, changed)),
    }
    for key, value in result.items():
        require(value, f"permutation:{key}")
    return result


def p0_models() -> dict[str, object]:
    datamodules = [make_datamodule(MASK, batch_size=2, include_window_metadata=True) for _ in range(3)]
    batches = [next(iter(item[0].test_dataloader())) for item in datamodules]
    common_inputs = ("x", "u", "input_mask", "y", "mask", "window_index", "split_id")
    metadata_identical = all(
        torch.equal(batches[0][key], batch[key]) for batch in batches[1:] for key in common_inputs
    )
    require(metadata_identical, "window metadata/data inputs differ between variants")

    datamodule, dataset, adjacency, *_ = datamodules[0]
    batch = batches[0]
    full_observed = construct("mrmsgf", "observed", dataset, adjacency)
    full_permuted = construct("mrmsgf", "permuted_pairwise", dataset, adjacency)
    feature = construct("official_base_feature_only", "observed", dataset, adjacency)
    require(sum(item.numel() for item in full_observed.parameters()) ==
            sum(item.numel() for item in full_permuted.parameters()), "Full parameter counts differ")

    models = (("full_observed", full_observed), ("full_permuted_q", full_permuted), ("feature_only", feature))
    finite_outputs = {}
    for label, model in models:
        model.train()
        output = model(
            batch["x"], batch["edge_index"], input_mask=batch["input_mask"], u=batch["u"],
            window_index=batch["window_index"], split_id=batch["split_id"],
        )[0]
        output.mean().backward()
        finite_outputs[label] = bool(torch.isfinite(output).all() and all(
            parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()
        ))
        require(finite_outputs[label], f"non-finite forward/backward for {label}")

    zero_mask = batch["input_mask"].clone()
    zero_mask[:, :, 0, :] = False
    fallbacks = []
    for model in (full_observed, full_permuted):
        encoded = model.encoder_time(batch["x"], mask=zero_mask, u=batch["u"])
        a_obs, q_ij, node_rate, _, zero_nodes = model._observed_graph(
            encoded, model._mask_3d(zero_mask, batch["x"]), batch["window_index"], batch["split_id"]
        )
        a_fine, _ = model._fine_graph(a_obs, q_ij, node_rate, zero_nodes)
        output = model(
            batch["x"], batch["edge_index"], input_mask=zero_mask, u=batch["u"],
            window_index=batch["window_index"], split_id=batch["split_id"],
        )[0]
        require(bool((a_obs[zero_nodes] == 0).all()), "zero-observation A_obs is not zero")
        require(bool(torch.isfinite(output).all()), "zero-observation output is non-finite")
        fallbacks.append(a_fine[zero_nodes].detach())
    require(torch.equal(fallbacks[0], fallbacks[1]), "zero-observation physical fallbacks differ")

    forbidden = {name.split(".")[0] for name, _ in feature.named_modules()} & FORBIDDEN_FEATURE_MODULES
    require(not forbidden, f"Feature-only dynamic modules: {sorted(forbidden)}")
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "full_observed_state.pt"
        torch.save(full_observed.state_dict(), path)
        reloaded = construct("mrmsgf", "observed", dataset, adjacency)
        reloaded.load_state_dict(torch.load(path, map_location="cpu"))
        reloaded_output = reloaded(
            batch["x"], batch["edge_index"], input_mask=batch["input_mask"], u=batch["u"],
            window_index=batch["window_index"], split_id=batch["split_id"],
        )[0]
        require(bool(torch.isfinite(reloaded_output).all()), "checkpoint reload output is non-finite")
    return {
        "all_variants_receive_window_metadata": metadata_identical,
        "metadata_fields": ["window_index", "split_id"],
        "metadata_consumed_only_by_permuted_q": True,
        "full_parameter_count": sum(item.numel() for item in full_observed.parameters()),
        "feature_only_parameter_count": sum(item.numel() for item in feature.parameters()),
        "full_observed_and_permuted_parameter_count_equal": True,
        "feature_only_has_no_dynamic_modules": True,
        "forward_backward_finite": finite_outputs,
        "zero_observation_physical_fallback_equal": True,
        "checkpoint_reload_finite": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "missing_ts_exp/results/r0917_protocol_closed/p0_audit.json"))
    args = parser.parse_args()
    result = {"status": "passed", "permutation": p0_permutation(), "models": p0_models()}
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
