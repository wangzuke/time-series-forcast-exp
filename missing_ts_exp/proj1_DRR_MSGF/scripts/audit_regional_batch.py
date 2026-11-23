"""Exercise the planned regional variants on one CPU batch before GPU runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from ..src.training.run_mrmsgf_official_controlled import PROJECT_ROOT, build_predictor, make_datamodule


def audit(config: dict, mask: Path, assignment: Path, stage: str, out: Path) -> None:
    dm, windows, adjacency, *_ = make_datamodule(mask, batch_size=2)
    predictor = build_predictor(SimpleNamespace(
        model="mrmsgf", mr_variant="full", graph_topk=config["graph_topk"],
        reliability_beta=config["reliability_beta"], fusion_mode=config["fusion_mode"],
        physical_anchor_lambda=config["physical_anchor_lambda"],
        dynamic_region_temperature=config["dynamic_region_temperature"],
        region_assignment_path=assignment, disable_overlap_reliability=False,
        disable_readout_reliability=False,
    ), windows, adjacency)
    model = predictor.model
    model.inspect_regions = True
    batch = next(iter(dm.val_dataloader()))
    modes = config["p2_orders"] if stage == "p2" else ["physical_static", "physical_anchored_dynamic"]
    summary, fine_graphs, fine_states = {}, [], []
    for mode in modes:
        if stage == "p2":
            model.regional_order = mode
        else:
            model.region_mode = mode
        state = []
        original = model._fine_propagate

        def record_fine(*args):
            result = original(*args)
            state.append(result.detach().clone())
            return result

        model._fine_propagate = record_fine
        try:
            prediction, *_ = predictor.predict_batch(batch)
        finally:
            del model._fine_propagate
        predictor.zero_grad(set_to_none=True)
        prediction.square().mean().backward()
        diag = model.last_diagnostics
        for name in (*config["p2_orders"], "fine_adjacency", "region_assignment"):
            matrix = diag[name]
            if not (torch.isfinite(matrix).all() and (matrix >= 0).all()
                    and torch.allclose(matrix.sum(-1), torch.ones_like(matrix[..., 0]), atol=1e-5)):
                raise RuntimeError(f"{mode}: invalid {name}")
        if not torch.isfinite(prediction).all() or not all(
                par.grad is None or torch.isfinite(par.grad).all() for par in predictor.parameters()):
            raise RuntimeError(f"{mode}: nonfinite prediction or gradient")
        if model.edge_gate[0].weight.grad is None:
            raise RuntimeError(f"{mode}: edge gate lost its gradient")
        fine_graphs.append(diag["fine_adjacency"].clone())
        fine_states.append(torch.stack(state))
        difference = float(torch.linalg.matrix_norm(diag[config["p2_orders"][0]]
                                                   - diag[config["p2_orders"][1]]).mean())
        summary[mode] = {"mean_coarse_graph_distance": difference}
    if not (torch.equal(fine_graphs[0], fine_graphs[1]) and torch.equal(fine_states[0], fine_states[1])):
        raise RuntimeError("The regional comparison unexpectedly changed the fine graph or features.")
    if stage == "p2" and not all(v["mean_coarse_graph_distance"] > 1e-6 for v in summary.values()):
        raise RuntimeError("The coarse graphs are indistinguishable on the audit batch.")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"stage": stage, "passed": True, "variants": summary}, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=["p2", "p3"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    seed = config["development_seed"]
    audit(config, PROJECT_ROOT / config["mask_template"].format(seed=seed),
          PROJECT_ROOT / config["physical_assignment"], args.stage, args.out)


if __name__ == "__main__":
    main()
