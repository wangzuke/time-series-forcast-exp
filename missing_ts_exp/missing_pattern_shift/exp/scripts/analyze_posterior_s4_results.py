#!/usr/bin/env python3
"""Audit and analyze the completed fixed suite without inference or retraining.

Training seeds are the repeat unit. Test-mask seeds are averaged within each
checkpoint; no independence assumption is made about overlapping windows.
"""
from __future__ import annotations

import csv
import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SUITE = "posterior_s4_r50_b24_l96_h96_1119"
METRICS = ("mse_standardized", "mae_standardized", "mse_raw", "mae_raw")
FAMILIES = ("P", "B", "B-uniform")
VARIANTS = ("LI-S4", "STAT-S4", "PU-S4")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    torch.set_num_threads(2)
    manifest = read(ROOT / "configs" / f"{SUITE}.json")
    datasets = sorted({job["dataset"] for job in manifest["jobs"]})
    output = ROOT / "summary" / SUITE / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    protocols, training, probes = {}, {}, {}
    mask_ids, source_ids = {}, []
    for job in manifest["jobs"]:
        directory = ROOT / "results" / SUITE / job["id"]
        protocol = read(directory / "protocol.json")
        completed = read(directory / "training_complete.json")
        progress = read(directory / "progress.json")
        assert completed["status"] == progress["status"] == "complete"
        assert completed["fingerprint"] == progress["fingerprint"] == protocol["fingerprint"]
        assert protocol["source"] == manifest["source"]
        assert not (directory / "failure.json").exists()
        source_ids.append(protocol["fingerprint"])
        key = (job["dataset"], job["variant"], job["family"], job["seed"])
        protocols[key], training[key] = protocol, completed
        probes[key] = read(directory / "input_probe.json")
        assert probes[key]["hidden_value_change_max_abs"] == 0
        for suffix in ("best", "last"):
            path = ROOT / "checkpoints" / SUITE / f"{job['id']}.{suffix}.pt"
            assert path.stat().st_size > 0
            checkpoint = torch.load(path, map_location="cpu")
            assert checkpoint["fingerprint"] == protocol["fingerprint"]
            assert all(torch.isfinite(value).all() for value in checkpoint["model"].values())
            if suffix == "last":
                assert "optimizer" in checkpoint and "rng" in checkpoint
            del checkpoint
        for family in FAMILIES:
            for seed in (9101, 9102):
                path = directory / f"test_{family}_m{seed}.json"
                record = read(path)
                assert record["status"] == "complete" and record["fingerprint"] == protocol["fingerprint"]
                assert all(math.isfinite(record[k]) for k in METRICS)
                mask_key = (job["dataset"], family, seed)
                if mask_key in mask_ids:
                    assert mask_ids[mask_key] == record["mask_fingerprint"]
                mask_ids[mask_key] = record["mask_fingerprint"]
                with np.load(path.with_suffix(".npz")) as arrays:
                    assert len(arrays["index"]) == record["n_windows"]
                    assert np.isfinite(arrays["window_mse"]).all()
                    assert abs(float(arrays["window_mse"].mean()) - record["mse_standardized"]) < 2e-6
                results[key + (family, seed)] = record
    # Files in the evidence set must still correspond to the frozen manifest.
    for relative, sha in manifest["source"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == sha

    def records(data, variant, train, test, seed=None):
        return [results[(data, variant, train, se, test, mask)]
                for se in ((seed,) if seed is not None else (2024, 2025)) for mask in (9101, 9102)]

    def avg(data, variant, train, test, metric="mse_standardized", seed=None):
        return np.mean([r[metric] for r in records(data, variant, train, test, seed)], axis=0)

    seed_cells, cells, comparisons, mismatches, diagnostics, jobs = [], [], [], [], [], []
    for data in datasets:
        for variant in VARIANTS:
            for train in ("P", "B"):
                for seed in (2024, 2025):
                    key = (data, variant, train, seed)
                    run, protocol = training[key], protocols[key]
                    best = min(run["history"], key=lambda x: x["val_mse"])
                    jobs.append(dict(dataset=data, variant=variant, train=train, seed=seed,
                                     parameters=protocol["parameter_counts"]["total"], best_epoch=best["epoch"],
                                     epochs=len(run["history"]), best_val_mse=run["best_val_mse"],
                                     train_mse_at_best=best["train"]["mse"],
                                     train_seconds=sum(x["train"]["seconds"] for x in run["history"]),
                                     response_mean_abs=probes[key]["value_perturbation_mean_abs"]))
                for test in FAMILIES:
                    row = dict(dataset=data, variant=variant, train=train, test=test)
                    for seed in (2024, 2025):
                        seed_cells.append(dict(**row, seed=seed, **{m: float(avg(data, variant, train, test, m, seed)) for m in METRICS}))
                    for metric in METRICS:
                        values = [float(avg(data, variant, train, test, metric, seed)) for seed in (2024, 2025)]
                        row[metric] = float(np.mean(values))
                        row[metric + "_seed_sd"] = float(np.std(values, ddof=1))
                    cells.append(row)
                    if variant != "LI-S4" and test != "B-uniform":
                        for seed in (2024, 2025):
                            diag = results[(data, variant, train, seed, test, 9101)]["posterior_diagnostic"]
                            diagnostics.append(dict(dataset=data, variant=variant, train=train, test=test, seed=seed,
                                                    hidden_mse=diag["hidden_mse"], mean_variance=diag["mean_predictive_variance"],
                                                    coverage_90=diag["coverage_90"],
                                                    **{f"quartile{i+1}_mse": v["mse"] for i, v in enumerate(diag["variance_quartiles"])}))
            for test in ("B", "P"):
                matched_train, shifted_train = ("B", "P") if test == "B" else ("P", "B")
                values = [100 * (avg(data, variant, shifted_train, test, seed=seed) /
                                 avg(data, variant, matched_train, test, seed=seed) - 1) for seed in (2024, 2025)]
                mismatches.append(dict(dataset=data, variant=variant, fixed_test=test,
                                       cost_percent=float(np.mean(values)), seed2024=float(values[0]), seed2025=float(values[1])))
        for train in ("P", "B"):
            for test in FAMILIES:
                for baseline in ("LI-S4", "STAT-S4"):
                    channel_p = avg(data, "PU-S4", train, test, "mse_by_channel")
                    channel_b = avg(data, baseline, train, test, "mse_by_channel")
                    horizon_p = avg(data, "PU-S4", train, test, "mse_by_horizon")
                    horizon_b = avg(data, baseline, train, test, "mse_by_horizon")
                    row = dict(dataset=data, train=train, test=test, baseline=baseline,
                               improved_channels=int((channel_p < channel_b).sum()), channels=len(channel_p),
                               horizon1_24_delta_percent=float(100 * (horizon_p[:24].mean() / horizon_b[:24].mean() - 1)),
                               horizon25_96_delta_percent=float(100 * (horizon_p[24:].mean() / horizon_b[24:].mean() - 1)))
                    for metric in METRICS:
                        row[metric + "_delta_percent"] = float(100 * (avg(data, "PU-S4", train, test, metric) / avg(data, baseline, train, test, metric) - 1))
                    for seed in (2024, 2025):
                        row[f"mse_delta_seed{seed}_percent"] = float(100 * (avg(data, "PU-S4", train, test, seed=seed) / avg(data, baseline, train, test, seed=seed) - 1))
                    comparisons.append(row)
    audits = []
    for data in datasets:
        for family in FAMILIES:
            values = [results[(data, "PU-S4", "P", 2024, family, seed)]["mask_audit"] for seed in (9101, 9102)]
            audits.append(dict(dataset=data, test=family,
                               last_observed_position=float(np.mean([v["last_observed_position_mean"] for v in values])),
                               adjacent_channel_joint_missing=float(np.mean([v["adjacent_channel_joint_missing_fraction"] for v in values]))))
    for name, rows in (("cells", cells), ("seed_cells", seed_cells), ("comparisons", comparisons),
                       ("mismatch_costs", mismatches), ("uncertainty", diagnostics), ("training", jobs), ("mask_audits", audits)):
        write_csv(output / f"{name}.csv", rows)
    analysis = {"suite": SUITE, "audit": {"jobs": len(manifest["jobs"]), "evaluations": len(results),
                 "checkpoint_files": 2 * len(manifest["jobs"]), "checkpoints_deserialized_on_cpu": True,
                 "errors": [], "paired_test_masks_equal": True},
                "aggregation": "average mask seeds within checkpoint, then summarize two training seeds; no significance claim",
                "cells": cells, "seed_cells": seed_cells, "comparisons": comparisons,
                "mismatch_costs": mismatches, "uncertainty": diagnostics, "training": jobs, "mask_audits": audits}
    (output / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"audit": analysis["audit"], "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default=SUITE)
    SUITE = parser.parse_args().suite
    main()
