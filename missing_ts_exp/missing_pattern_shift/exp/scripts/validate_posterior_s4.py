#!/usr/bin/env python3
"""Numerical reference checks, synthetic oracle, GPU smoke and benchmarks."""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from protocol import EXP_ROOT, atomic_json_dump, per_channel_point_mask, per_channel_block_mask
from posterior_s4_common import (SUITE, ExperimentDataset, ModelConfig, PosteriorS4, make_optimizer,
                                 mask_fingerprint, memory_stats, move_batch, runtime_metadata, seed_all, source_fingerprints)
from posterior_s4 import GaussianObservation, interpolate_history, s4_module


def numerical_checks(device):
    torch.set_num_threads(2)
    seed_all(12)
    cfg = ModelConfig(channels=3, seq_len=6, rank=2)
    reference = GaussianObservation(cfg).double()
    with torch.no_grad():
        reference.loadings.mul_(3)
        reference.mean.copy_(torch.tensor([.3, -.2, .6]))
    x = torch.randn(4, 6, 3, dtype=torch.float64)
    mask = (torch.rand(4, 6, 3) > .4).double()
    mask[0] = 0
    mask[1, :2] = 0
    mask[2] = 1
    _, nll, details = reference(x, mask, return_details=True)
    ref_means, ref_covs, ref_nll = [], [], []
    for b in range(4):
        rows_m, rows_s = [], []
        for t in range(6):
            visible = mask[b, :t + 1].bool().reshape(-1)
            A = reference.loadings[:t + 1].reshape(-1, 2)[visible]
            noise = reference.variance.expand(t + 1, 3).reshape(-1)[visible]
            residual = (x[b, :t + 1] - reference.mean).reshape(-1)[visible]
            # Independent covariance-space reference, not the information
            # formula used in the actual module.
            cov_x = A @ A.T + torch.diag(noise)
            cov = torch.eye(2, dtype=torch.float64) - A.T @ torch.linalg.solve(cov_x, A)
            mean = A.T @ torch.linalg.solve(cov_x, residual)
            rows_m.append(mean)
            rows_s.append(cov)
            if t == 5:
                value = .5 * (len(residual) * math.log(2 * math.pi) + torch.linalg.slogdet(cov_x)[1]
                              + residual @ torch.linalg.solve(cov_x, residual)) / max(len(residual), 1)
                ref_nll.append(value)
        ref_means.append(torch.stack(rows_m))
        ref_covs.append(torch.stack(rows_s))
    expected_m, expected_s = torch.stack(ref_means), torch.stack(ref_covs)
    errors = {"mean_float64": float((details["mean"] - expected_m).abs().max()),
              "covariance_float64": float((details["covariance"] - expected_s).abs().max()),
              "nll_float64": float((nll - torch.stack(ref_nll)).abs().max())}
    assert max(errors.values()) < 1e-6, errors
    updated = torch.where(mask.bool(), x, torch.full_like(x, float("nan")))
    f1, n1 = reference(x, mask)
    f2, n2 = reference(updated, mask)
    assert torch.equal(f1, f2) and torch.equal(n1, n2)
    covariance_drop = details["covariance"][:, :-1] - details["covariance"][:, 1:]
    assert torch.linalg.eigvalsh(covariance_drop).min() > -1e-10
    assert torch.allclose(details["mean"][0], torch.zeros_like(details["mean"][0]))
    assert torch.allclose(details["covariance"][0], torch.eye(2, dtype=torch.float64).expand(6, 2, 2))
    # Differentiate through both the posterior representation and likelihood.
    (f1.square().mean() + n1.mean()).backward()
    for p in reference.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    gpu = GaussianObservation(cfg).to(device)
    gpu.load_state_dict(reference.float().state_dict())
    _, gnll, gd = gpu(x.float().to(device), mask.float().to(device), return_details=True)
    errors.update(mean_float32=float((gd["mean"].cpu().double() - expected_m.detach()).abs().max()),
                  covariance_float32=float((gd["covariance"].cpu().double() - expected_s.detach()).abs().max()),
                  nll_float32=float((gnll.cpu().double() - torch.stack(ref_nll).detach()).abs().max()))
    assert max(errors[k] for k in errors if "float32" in k) < 1e-4, errors
    filled = interpolate_history(x.float().to(device), mask.float().to(device))
    assert torch.isfinite(filled).all()
    assert torch.equal(filled[mask.bool().to(device)], x.float().to(device)[mask.bool().to(device)])
    return {"status": "passed", "max_absolute_errors": errors, "missing_value_invariance": True,
            "prior_boundary": True, "covariance_monotonicity": True, "gradient_check": True}


def oracle_checks(device):
    report = []
    length, channels, rank, samples = 96, 12, 8, 2048
    for data_seed in (7201, 7202, 7203):
        rng = np.random.default_rng(data_seed)
        time_axis = np.arange(192)[:, None]
        frequency = rng.uniform(.5, 4, (1, rank))
        basis = np.cos(2 * np.pi * time_axis * frequency / 96 + rng.uniform(0, 2 * np.pi, (1, rank)))
        loadings = basis[:, None, :] * rng.normal(size=(1, channels, rank)) / math.sqrt(rank)
        z = rng.normal(size=(samples, rank))
        values = np.einsum("br,lcr->blc", z, loadings).astype(np.float32)
        values[:, :96] += rng.normal(scale=.2, size=(samples, length, channels))
        values[:, 96:] += rng.normal(scale=.1, size=(samples, length, channels))
        obs = GaussianObservation(ModelConfig(channels=channels, rank=rank)).to(device)
        with torch.no_grad():
            obs.loadings.copy_(torch.as_tensor(loadings[:96], device=device))
            obs.raw_variance.fill_(math.log(math.expm1(.04 - .01)))
        future_a = torch.as_tensor(loadings[96:], dtype=torch.float32, device=device)
        for family in ("P", "B"):
            for mask_seed in (9101, 9102):
                rng_mask = np.random.default_rng(mask_seed)
                metrics = torch.zeros(3, dtype=torch.float64, device=device)
                for start in range(0, samples, 64):
                    count = min(64, samples - start)
                    generator = per_channel_point_mask if family == "P" else per_channel_block_mask
                    masks = np.stack([generator(length, channels, .5, rng_mask) if family == "P" else
                                      generator(length, channels, .5, 24, rng_mask) for _ in range(count)])
                    x = torch.as_tensor(values[start:start + count, :96], device=device)
                    mask = torch.as_tensor(masks, device=device)
                    with torch.no_grad():
                        _, _, d = obs(x, mask, return_details=True)
                        mean = torch.einsum("lcr,br->blc", future_a, d["mean"][:, -1])
                        var = torch.einsum("lci,bij,lcj->blc", future_a, d["covariance"][:, -1], future_a) + .01
                        target = torch.as_tensor(values[start:start + count, 96:], device=device)
                        error = (mean - target).square()
                        metrics += torch.stack((error.sum(), var.sum(), (error <= 1.6448536269514722**2 * var).sum())).double()
                mse, risk, coverage = (metrics / (samples * 96 * channels)).cpu().tolist()
                assert .87 < coverage < .93, (data_seed, family, coverage)
                assert abs(mse / risk - 1) < .12, (mse, risk)
                report.append(dict(data_seed=data_seed, family=family, mask_seed=mask_seed,
                                   mse=mse, theoretical_risk=risk, coverage_90=coverage))
    return report


def gpu_smoke(device):
    cfg = ModelConfig(channels=7)
    batch = {"x": torch.randn(8, 96, 7, device=device),
             "mask": (torch.rand(8, 96, 7, device=device) > .5).float()}
    common = None
    counts = {}
    for variant in ("LI-S4", "STAT-S4", "PU-S4"):
        model = PosteriorS4(variant, cfg, 2024).to(device)
        model.eval()
        current = {k: v.detach().cpu() for k, v in model.state_dict().items()
                   if k.startswith(("blocks.", "time_head.", "channel_head."))}
        if common is not None:
            assert all(torch.equal(v, common[k]) for k, v in current.items())
        common = current
        pred, nll = model(**batch)
        assert pred.shape == (8, 96, 7) and torch.isfinite(pred).all()
        old = pred.detach().clone()
        (pred.square().mean() + .1 * nll.mean()).backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(state)
        assert torch.equal(model(**batch)[0], old)
        changed = model(batch["x"] + batch["mask"] * .1, batch["mask"])[0]
        assert float((changed - old).abs().max()) > 1e-6
        counts[variant] = model.parameter_counts()
        if variant == "PU-S4":
            module = s4_module()
            original = module.cauchy_cuda
            calls = []
            def counted(*args, **kwargs):
                calls.append(True)
                return original(*args, **kwargs)
            module.cauchy_cuda = counted
            try:
                with torch.autograd.profiler.profile(use_cuda=True) as prof:
                    model(**batch)[0].square().mean().backward()
                names = sorted({event.name for event in prof.function_events if "cauchy" in event.name.lower()})
            finally:
                module.cauchy_cuda = original
            assert calls and names, (len(calls), names)
    ds = ExperimentDataset("Electricity", "train", "B", 2024, True)
    assert mask_fingerprint(ds, 1) != mask_fingerprint(ds, 2)
    audits = {}
    for family in ("P", "B", "B-uniform"):
        audit_ds = ExperimentDataset("Electricity", "test", family, 9101)
        for i in (0, 17, len(audit_ds) - 1):
            mask = audit_ds.mask_for(i)
            assert ((mask == 0).sum(0) == 48).all()
            if family != "P":
                for c in range(mask.shape[1]):
                    edges = np.flatnonzero(np.diff(np.r_[1, mask[:, c], 1]))
                    assert np.array_equal(edges[1::2] - edges[::2], [24, 24])
        audits[family] = audit_ds.audit(32)
    return {"status": "passed", "parameter_counts": counts, "cuda_cauchy_calls": len(calls),
            "profiler_events": names, "mask_audits": audits, "dynamic_epoch_masks": True,
            "common_initialization_equal": True, "checkpoint_roundtrip": True}


def benchmark(args, device):
    dataset = ExperimentDataset("Electricity", "train", "P", 2024, True)
    dataset.set_epoch(1)
    cfg = ModelConfig(channels=dataset.n_features)
    records = []
    for variant in args.variants:
        model = PosteriorS4(variant, cfg, 2024).to(device)
        optimizer, _ = make_optimizer(model)
        fixed = move_batch(torch.utils.data.default_collate([dataset[i] for i in range(args.batch_size)]), device)
        accumulation = 64 // args.batch_size
        def update(batches):
            optimizer.zero_grad(set_to_none=True)
            for batch in batches:
                p, n = model(batch["x"], batch["mask"])
                ((p - batch["y"]).square().mean() + .1 * n.mean()).div(accumulation).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
        for mode in ("device", "end_to_end"):
            times = []
            peaks = []
            for repeat in range(args.repeats):
                loader = iter(DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                                         pin_memory=True, persistent_workers=False)) if mode == "end_to_end" else None
                for _ in range(args.warmup):
                    update([move_batch(next(loader), device) for _ in range(accumulation)] if loader else [fixed] * accumulation)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                begin = time.perf_counter()
                for _ in range(args.steps):
                    update([move_batch(next(loader), device) for _ in range(accumulation)] if loader else [fixed] * accumulation)
                torch.cuda.synchronize()
                seconds = time.perf_counter() - begin
                times.append(seconds)
                peaks.append(memory_stats())
                del loader
            records.append({"variant": variant, "mode": mode, "microbatch": args.batch_size,
                            "effective_batch": 64, "workers": args.workers, "steps": args.steps,
                            "seconds": times, "samples_per_second": 64 * args.steps / float(np.median(times)), "memory": peaks})
            print(json.dumps(records[-1]), flush=True)
        del model, optimizer, fixed
        torch.cuda.empty_cache()
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("check", "benchmark"), default="check")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--variants", nargs="+", default=["LI-S4", "PU-S4"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(2)
    seed_all(2024)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA access is required")
    device = torch.device("cuda:0")
    if args.mode == "check":
        result = {"status": "passed", "numerical": numerical_checks(device),
                  "gpu": gpu_smoke(device), "oracle": oracle_checks(device), "runtime": runtime_metadata(),
                  "source": source_fingerprints()}
    else:
        result = {"status": "complete", "benchmarks": benchmark(args, device), "runtime": runtime_metadata()}
    atomic_json_dump(result, args.output)
    print(json.dumps({"status": result["status"], "output": str(args.output)}), flush=True)
