"""Inspect regional graphs and validate dynamic memberships on a frozen model."""
from __future__ import annotations

import argparse
import json
from itertools import islice
from pathlib import Path

import numpy as np
import torch

from .regional_checkpoint import load_checkpoint


def quantiles(values) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {}
    return dict(mean=float(values.mean()), p50=float(np.quantile(values, 0.50)),
                p95=float(np.quantile(values, 0.95)), minimum=float(values.min()),
                maximum=float(values.max()))


@torch.no_grad()
def diagnose(config: dict, checkpoint_result: Path, out: Path, device: str) -> dict:
    row, dm, predictor = load_checkpoint(checkpoint_result, config["formal"]["batch_size"], device)
    model = predictor.model
    model.inspect_regions = True
    out.mkdir(parents=True, exist_ok=True)

    # Freeze a matched soft control using equally spaced training batches only.
    train = dm.train_dataloader(shuffle=False)
    indices = set(np.linspace(0, len(train) - 1,
                              min(len(train), config["static_soft_training_batches"]), dtype=int))
    total = torch.zeros_like(model.region_assignment, dtype=torch.float64)
    seen = 0
    for index, batch in enumerate(train):
        if index in indices:
            predictor.predict_batch(batch.to(device))
            assignment = model.last_diagnostics["dynamic_assignment"]
            total += assignment.to(torch.float64).sum(dim=0)
            seen += assignment.size(0)
    if not seen:
        raise RuntimeError("No training windows available for the fixed soft assignment.")
    fixed = (total / seen).to(model.region_assignment.dtype)
    if torch.isfinite(fixed).all():
        np.save(out / "fixed_soft.npy", fixed.cpu().numpy())

    keys = ("graph_difference", "ftr_density", "ctf_density", "entropy", "drift", "frobenius_drift",
            "low_observation_drift", "hard_departure", "fixed_soft_departure",
            "low_obs_entropy", "high_obs_entropy", "short_gap_entropy", "long_gap_entropy")
    values: dict[str, list[np.ndarray]] = {key: [] for key in keys}

    def add(key, value):
        values[key].append(value.detach().flatten().cpu().numpy())

    previous = None
    previous_start = None
    finite = bool(torch.isfinite(fixed).all())
    row_stochastic = True
    minimum_mass = torch.full((fixed.size(1),), float("inf"), device=device)
    mean_mass = torch.zeros_like(minimum_mass)
    windows = 0
    for batch in islice(dm.val_dataloader(), config["validation_diagnostic_batches"]):
        batch = batch.to(device)
        predictor.predict_batch(batch)
        diag = model.last_diagnostics
        ftr = diag["fuse_then_coarsen"]
        ctf = diag["coarsen_then_fuse"]
        dynamic = diag["dynamic_assignment"]
        add("graph_difference", torch.linalg.matrix_norm(ftr - ctf))
        add("ftr_density", (ftr > 1e-8).float().mean(dim=(1, 2)))
        add("ctf_density", (ctf > 1e-8).float().mean(dim=(1, 2)))
        finite &= all(bool(torch.isfinite(x).all()) for x in (ftr, ctf, dynamic))
        row_stochastic &= all(bool(torch.allclose(x.sum(-1), torch.ones_like(x[..., 0]),
                                                   atol=1e-5)) for x in (ftr, ctf, dynamic))
        entropy = -(dynamic * dynamic.clamp_min(1e-12).log()).sum(-1)
        obs, gap = diag["node_observation_rate"], diag["block_fraction"]
        add("entropy", entropy)
        add("low_obs_entropy", entropy[obs <= 0.25])
        add("high_obs_entropy", entropy[obs >= 0.75])
        add("short_gap_entropy", entropy[gap <= 0.25])
        add("long_gap_entropy", entropy[gap >= 0.50])
        add("hard_departure", 0.5 * (dynamic - model.region_assignment).abs().sum(-1))
        add("fixed_soft_departure", 0.5 * (dynamic - fixed).abs().sum(-1))
        mass = dynamic.sum(dim=1)
        minimum_mass = torch.minimum(minimum_mass, mass.amin(dim=0))
        mean_mass += mass.sum(dim=0)
        windows += dynamic.size(0)
        current_start = batch.input["window_index"][:, 0, 0, 0]
        old = dynamic[:-1] if previous is None else torch.cat([previous, dynamic[:-1]])
        old_start = current_start[:-1] if previous_start is None else torch.cat([previous_start, current_start[:-1]])
        new = dynamic[-len(old):] if len(old) else dynamic[:0]
        starts = current_start[-len(old):] if len(old) else current_start[:0]
        consecutive = starts - old_start == 1
        drift = 0.5 * (new - old).abs().sum(-1)
        aligned_obs = obs[-len(old):] if len(old) else obs[:0]
        add("drift", drift[consecutive])
        add("frobenius_drift", torch.linalg.matrix_norm(new - old)[consecutive])
        add("low_observation_drift", drift[consecutive][aligned_obs[consecutive] <= 0.25])
        previous, previous_start = dynamic[-1:], current_start[-1:]

    stats = {key: quantiles(np.concatenate(items)) for key, items in values.items() if items}
    for key in ("graph_difference", "ftr_density", "ctf_density"):
        np.save(out / f"{key}.npy", np.concatenate(values[key]))
    typical_mass = fixed.size(0) / fixed.size(1)
    mass_fraction = float(minimum_mass.min()) / typical_mass
    p95 = max(stats.get(key, {}).get("p95", 0.0) for key in ("drift", "low_observation_drift"))
    thresholds_pass = (
        finite and row_stochastic and windows > 0
        and mass_fraction >= config["min_region_mass_fraction"]
        and p95 <= config["max_adjacent_total_variation_p95"]
        and stats.get("fixed_soft_departure", {}).get("mean", 0.0)
        >= config["min_dynamic_departure_from_static_soft"]
    )
    output = dict(
        result_json=str(checkpoint_result), checkpoint=row["best_checkpoint"],
        selection_split="validation", validation_windows=windows,
        fixed_soft_from_training_windows=seen, finite=finite, row_stochastic=row_stochastic,
        minimum_region_mass_fraction=mass_fraction, per_region_mean_mass=(mean_mass / windows).tolist(),
        effective_regions_per_node=float(np.exp(stats["entropy"]["mean"])),
        thresholds=config | {"regional_order": row["regional_order"]},
        stats=stats, passed=bool(thresholds_pass),
    )
    (out / "regional_diagnostics.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), layout="constrained")
    axes[0].hist(np.concatenate(values["graph_difference"]), bins=30)
    axes[0].set(xlabel="Frobenius distance between regional graphs", ylabel="Windows")
    axes[1].hist([np.concatenate(values[k]) for k in ("ftr_density", "ctf_density")],
                 bins=20, label=["Fuse then coarsen", "Coarsen then fuse"])
    axes[1].set(xlabel="Proportion of nonzero regional edges", ylabel="Windows")
    axes[1].legend()
    fig.savefig(out / "p2_graph_comparison.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), layout="constrained")
    groups = ("low_obs_entropy", "high_obs_entropy", "short_gap_entropy", "long_gap_entropy")
    axes[0].boxplot([np.concatenate(values[k]) for k in groups],
                    labels=["Low obs.", "High obs.", "Short gap", "Long gap"], showfliers=False)
    axes[0].set_ylabel("Soft assignment entropy")
    axes[1].hist([model.region_assignment.sum(0).cpu().numpy(), fixed.sum(0).cpu().numpy(),
                  (mean_mass / windows).cpu().numpy()],
                 bins=12, label=["Physical", "Fixed soft", "Dynamic"])
    axes[1].set(xlabel="Region membership mass", ylabel="Regions")
    axes[1].legend()
    fig.savefig(out / "p3_assignment_diagnostics.png", dpi=160)
    plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    report = diagnose(json.loads(args.config.read_text()), args.result, args.out, args.device)
    print("Dynamic assignment diagnostic passed:", report["passed"])


if __name__ == "__main__":
    main()
