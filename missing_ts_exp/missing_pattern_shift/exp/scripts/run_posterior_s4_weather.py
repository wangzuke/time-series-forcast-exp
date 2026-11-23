#!/usr/bin/env python3
"""One resumable training run, followed by fixed P/B and B-uniform tests."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import tempfile
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader

from protocol import DATASETS, EXP_ROOT, atomic_json_dump, atomic_torch_save
from posterior_s4_common import (SUITE, ExperimentDataset, ModelConfig, PosteriorS4, VARIANTS, digest,
                                 make_optimizer, mask_audit, mask_fingerprint, memory_stats, move_batch,
                                 runtime_metadata, seed_all, sha_file, source_fingerprints as frozen_source_fingerprints)


# Deliberately retain the original trainer unchanged for its frozen evidence.
# This snapshot changes only dataset discovery, suite namespace and provenance.
SUITE = "posterior_s4_weather_r50_b24_l96_h96_1119"


def source_fingerprints():
    return {**frozen_source_fingerprints(),
            "scripts/run_posterior_s4_weather.py": sha_file(pathlib.Path(__file__))}


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


@torch.no_grad()
def evaluate(model, loader, device, mean, std, output=None):
    model.eval()
    channels = model.cfg.channels
    total = torch.zeros(5, dtype=torch.float64, device=device)
    channel_sq = torch.zeros(channels, dtype=torch.float64, device=device)
    step_sq = torch.zeros(96, dtype=torch.float64, device=device)
    per_window, indices = [], []
    diagnostic = []
    started = time.perf_counter()
    for batch in loader:
        batch = move_batch(batch, device)
        pred, _ = model(batch["x"], batch["mask"])
        error = pred - batch["y"]
        raw_error = pred * std + mean - batch["y_raw"]
        total += torch.stack((error.square().sum(), error.abs().sum(),
                              raw_error.square().sum(), raw_error.abs().sum(), batch["y"].square().sum())).double()
        channel_sq += error.square().sum((0, 1)).double()
        step_sq += error.square().sum((0, 2)).double()
        per_window.append(error.square().mean((1, 2)).cpu().numpy())
        indices.append(batch["index"].cpu().numpy())
        if output is not None and sum(x[0].shape[0] for x in diagnostic) < 32:
            take = min(8, pred.shape[0])
            diagnostic.append((pred[:take].cpu().numpy(), batch["index"][:take].cpu().numpy()))
    torch.cuda.synchronize()
    windows = len(loader.dataset)
    values = (total / (windows * 96 * channels)).cpu().tolist()
    if not all(np.isfinite(values)):
        raise FloatingPointError(f"non-finite evaluation: {values}")
    result = dict(zip(("mse_standardized", "mae_standardized", "mse_raw", "mae_raw", "train_mean_baseline_mse"), values))
    result["skill_vs_train_mean"] = 1 - result["mse_standardized"] / max(values[4], 1e-12)
    result.update(n_windows=windows, n_target_elements=windows * 96 * channels,
                  seconds=time.perf_counter() - started,
                  mse_by_channel=(channel_sq / (windows * 96)).cpu().tolist(),
                  mse_by_horizon=(step_sq / (windows * channels)).cpu().tolist())
    if output is not None:
        with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as handle:
            temporary = pathlib.Path(handle.name)
            try:
                np.savez_compressed(handle, index=np.concatenate(indices), window_mse=np.concatenate(per_window),
                                    example_predictions=np.concatenate([x[0] for x in diagnostic]),
                                    example_indices=np.concatenate([x[1] for x in diagnostic]))
                handle.flush()
                os.replace(temporary, output.with_suffix(".npz"))
            finally:
                temporary.unlink(missing_ok=True)
    return result


def train_epoch(model, loader, optimizer, args, device, progress, base_progress):
    model.train()
    sums = torch.zeros(4, device=device)
    count = 0
    group_count = 0
    accumulation = args.effective_batch // args.batch_size
    n_batches = len(loader)
    started = time.perf_counter()
    last_grad = None
    for number, batch in enumerate(loader):
        if number % accumulation == 0:
            optimizer.zero_grad(set_to_none=True)
            group_start = number * args.batch_size
            group_count = min(args.effective_batch, len(loader.dataset) - group_start)
        batch = move_batch(batch, device)
        pred, nll = model(batch["x"], batch["mask"])
        forecast = (pred - batch["y"]).square().mean()
        obs_loss = nll.mean()
        loss = forecast + args.aux_weight * obs_loss
        (loss * (pred.shape[0] / group_count)).backward()
        sums[:3] += torch.stack((forecast.detach(), obs_loss.detach(), loss.detach())) * pred.shape[0]
        count += pred.shape[0]
        if (number + 1) % accumulation == 0 or number + 1 == n_batches:
            last_grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            sums[3] += (last_grad > 1.0).to(sums.dtype)
            optimizer.step()
        if (number + 1) % 100 == 0:
            if not bool(torch.isfinite(sums).all()):
                raise FloatingPointError("non-finite training loss")
            info = {**base_progress, "batch": number + 1, "total_batches": n_batches,
                    "train_mse_so_far": float(sums[0] / count), "updated_at": time.time(),
                    "elapsed_epoch_seconds": time.perf_counter() - started}
            atomic_json_dump(info, progress)
            print(json.dumps(info), flush=True)
    torch.cuda.synchronize()
    values = (sums[:3] / count).cpu().tolist()
    return {"mse": values[0], "observation_nll": values[1], "total_loss": values[2],
            "clip_steps": int(sums[3].item()), "last_gradient_norm": float(last_grad),
            "samples": count, "seconds": time.perf_counter() - started}


def posterior_diagnostic(model, ds, device):
    if model.observation is None:
        return None
    selected = np.unique(np.linspace(0, len(ds) - 1, min(256, len(ds)), dtype=int))
    squared, variances, covered = [], [], []
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(selected), 16):
            chunk = selected[offset:offset + 16]
            batch = torch.utils.data.default_collate([ds[int(i)] for i in chunk])
            batch = move_batch(batch, device)
            _, _, details = model.observation(batch["x"], batch["mask"], return_details=True)
            obs = model.observation
            mu = obs.mean + torch.einsum("lcr,br->blc", obs.loadings, details["mean"][:, -1])
            var = torch.einsum("lci,bij,lcj->blc", obs.loadings, details["covariance"][:, -1], obs.loadings) + obs.variance
            # Truth is fetched only by this post-training evaluator.
            truth = torch.as_tensor(np.stack([ds.x[int(i):int(i) + 96] for i in chunk]), device=device)
            hidden = ~batch["mask"].bool()
            err = (mu - truth).square()[hidden]
            pv = var[hidden]
            squared.append(err.cpu().numpy())
            variances.append(pv.cpu().numpy())
            covered.append((err <= (1.6448536269514722**2) * pv).cpu().numpy())
    se, pv = np.concatenate(squared), np.concatenate(variances)
    cuts = np.quantile(pv, [0, .25, .5, .75, 1])
    groups = []
    for i in range(4):
        take = (pv >= cuts[i]) & ((pv <= cuts[i + 1]) if i == 3 else (pv < cuts[i + 1]))
        groups.append({"count": int(take.sum()), "mse": float(se[take].mean()) if take.any() else None,
                       "variance": float(pv[take].mean()) if take.any() else None})
    return {"n_windows": len(selected), "hidden_mse": float(se.mean()), "mean_predictive_variance": float(pv.mean()),
            "coverage_90": float(np.concatenate(covered).mean()), "variance_quartiles": groups,
            "interpretation": "hidden-history marginal diagnostic, not future interval calibration"}


@torch.no_grad()
def input_probe(model, ds, device):
    model.eval()
    samples = [ds[i] for i in range(min(16, len(ds)))]
    batch = move_batch(torch.utils.data.default_collate(samples), device)
    original = model(batch["x"], batch["mask"])[0]
    perturbed = model(batch["x"] + .1 * batch["mask"], batch["mask"])[0]
    hidden_changed = torch.where(batch["mask"].bool(), batch["x"], torch.full_like(batch["x"], 999))
    ignored = model(hidden_changed, batch["mask"])[0]
    alt_family = "B" if ds.missing_family == "P" else "P"
    alternate = ExperimentDataset(ds.dataset, ds.split, alt_family, ds.seed)
    alt_batch = move_batch(torch.utils.data.default_collate([alternate[i] for i in range(len(samples))]), device)
    alt_prediction = model(alt_batch["x"], alt_batch["mask"])[0]
    value_max = float((original - perturbed).abs().max())
    invariance = float((original - ignored).abs().max())
    result = {"value_perturbation_max_abs": value_max, "value_perturbation_mean_abs": float((original - perturbed).abs().mean()),
              "mask_change_prediction_mean_abs": float((original - alt_prediction).abs().mean()),
              "hidden_value_change_max_abs": invariance, "sample_count": len(samples)}
    if invariance != 0 or value_max < 1e-7:
        raise RuntimeError(f"Trained-model input probe failed: {result}")
    return result


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run with access to GPU 0 or 5")
    torch.set_num_threads(args.threads)
    seed_all(args.seed)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats()
    train = ExperimentDataset(args.dataset, "train", args.family, args.seed, True)
    val = ExperimentDataset(args.dataset, "val", args.family, args.seed + 10000)
    if args.smoke_windows:
        train.n_samples = min(len(train), args.smoke_windows)
        val.n_samples = min(len(val), args.smoke_windows)
    cfg = ModelConfig(channels=train.n_features)
    model = PosteriorS4(args.variant, cfg, args.seed).to(device)
    runtime = runtime_metadata()
    if not runtime["has_cuda_extension"]:
        raise RuntimeError("S4 CUDA extension not loaded; full suite will not use a slow fallback")
    optimizer, optim_meta = make_optimizer(model)
    directory = EXP_ROOT / "results" / SUITE / args.job_id
    directory.mkdir(parents=True, exist_ok=True)
    ckpt_dir = EXP_ROOT / "checkpoints" / SUITE
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / f"{args.job_id}.best.pt"
    last_path = ckpt_dir / f"{args.job_id}.last.pt"
    progress = directory / "progress.json"
    record = {"variant": args.variant, "dataset": args.dataset, "train_family": args.family, "train_seed": args.seed,
              "model": cfg.to_dict(), "batch": args.batch_size, "effective_batch": args.effective_batch,
              "workers": args.workers, "threads": args.threads, "epochs": args.epochs, "patience": args.patience,
              "smoke_windows": args.smoke_windows,
              "aux_weight": args.aux_weight, "source": source_fingerprints(), "data_sha256": sha_file(EXP_ROOT / "data" / DATASETS[args.dataset]["file"]),
              "protocol": "strict_separated_B24_r50_per_channel_v1", "optimizer_groups": optim_meta,
              "parameter_counts": model.parameter_counts(), "runtime": runtime,
              "optimizer": "AdamW", "optimizer_betas": [0.9, 0.999], "gradient_clip_norm": 1.0,
              "mask_epoch1": mask_fingerprint(train, 1), "mask_epoch2": mask_fingerprint(train, 2),
              "validation_mask": mask_fingerprint(val), "train_audit": mask_audit(train), "validation_audit": mask_audit(val)}
    identity = {key: value for key, value in record.items() if key != "runtime"}
    fingerprint = digest(identity)
    record["fingerprint"] = fingerprint
    protocol_path = directory / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text())["fingerprint"] != fingerprint:
        raise RuntimeError("Existing job configuration/source fingerprint differs; use a new job identity")
    atomic_json_dump(record, protocol_path)
    options = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, persistent_workers=False)
    if args.workers:
        options["prefetch_factor"] = 2
    val_loader = DataLoader(val, shuffle=False, **options)
    best, stale, start_epoch, history = float("inf"), 0, 1, []
    complete_path = directory / "training_complete.json"
    if complete_path.exists():
        summary = json.loads(complete_path.read_text())
        if summary.get("fingerprint") != fingerprint:
            raise RuntimeError("Training summary fingerprint mismatch")
    else:
        if last_path.exists():
            saved = torch.load(last_path, map_location=device)
            if saved["fingerprint"] != fingerprint:
                raise RuntimeError("Checkpoint fingerprint mismatch")
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            best, stale = saved["best"], saved["stale"]
            history, start_epoch = saved["history"], saved["epoch"] + 1
            restore_rng(saved["rng"])
        started = time.time()
        for epoch in range(start_epoch, args.epochs + 1):
            if stale >= args.patience:
                break
            train.set_epoch(epoch)
            generator = torch.Generator().manual_seed(args.seed + 100000 + epoch)
            loader = DataLoader(train, shuffle=True, generator=generator, **options)
            base = {"status": "training", "job_id": args.job_id, "epoch": epoch, "pid": os.getpid()}
            atomic_json_dump({**base, "updated_at": time.time()}, progress)
            stats = train_epoch(model, loader, optimizer, args, device, progress, base)
            val_result = evaluate(model, val_loader, device,
                                  torch.as_tensor(val.mean, device=device), torch.as_tensor(val.std, device=device))
            value = val_result["mse_standardized"]
            improved = value < best
            if improved:
                best, stale = value, 0
                atomic_torch_save({"model": model.state_dict(), "epoch": epoch, "val_mse": value,
                                   "fingerprint": fingerprint, "config": cfg.to_dict(), "variant": args.variant}, best_path)
            else:
                stale += 1
            history.append({"epoch": epoch, "train": stats, "val_mse": value, "improved": improved})
            atomic_torch_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                              "best": best, "stale": stale, "history": history, "rng": rng_state(),
                              "fingerprint": fingerprint}, last_path)
            print(json.dumps(history[-1]), flush=True)
        summary = {"status": "complete", "job_id": args.job_id, "fingerprint": fingerprint,
                   "best_val_mse": best, "history": history, "training_seconds_this_session": time.time() - started,
                   "stop_reason": "early_stopping" if stale >= args.patience else "max_epochs", **memory_stats()}
        atomic_json_dump(summary, complete_path)
    saved = torch.load(best_path, map_location=device)
    if saved["fingerprint"] != fingerprint:
        raise RuntimeError("Best checkpoint fingerprint mismatch")
    model.load_state_dict(saved["model"])
    atomic_json_dump(input_probe(model, val, device), directory / "input_probe.json")
    for family in ("P", "B", "B-uniform"):
        for seed in (9101, 9102):
            output = directory / f"test_{family}_m{seed}.json"
            if output.exists():
                existing = json.loads(output.read_text())
                if existing.get("fingerprint") == fingerprint and existing.get("status") == "complete":
                    continue
                raise RuntimeError(f"Unusable existing evaluation: {output}")
            atomic_json_dump({"status": "evaluating", "family": family, "mask_seed": seed,
                              "job_id": args.job_id, "updated_at": time.time()}, progress)
            ds = ExperimentDataset(args.dataset, "test", family, seed)
            if args.smoke_windows:
                ds.n_samples = min(len(ds), args.smoke_windows)
            result = evaluate(model, DataLoader(ds, shuffle=False, **options), device,
                              torch.as_tensor(ds.mean, device=device), torch.as_tensor(ds.std, device=device), output)
            result.update(status="complete", fingerprint=fingerprint, job_id=args.job_id,
                          test_family=family, test_mask_seed=seed, mask_fingerprint=mask_fingerprint(ds),
                          mask_audit=mask_audit(ds), best_epoch=saved["epoch"])
            if seed == 9101 and family in ("P", "B"):
                result["posterior_diagnostic"] = posterior_diagnostic(model, ds, device)
            atomic_json_dump(result, output)
            print(json.dumps({"test": family, "mask_seed": seed, "mse": result["mse_standardized"]}), flush=True)
    atomic_json_dump({"status": "complete", "job_id": args.job_id, "fingerprint": fingerprint,
                      "best_epoch": saved["epoch"], "updated_at": time.time(), **memory_stats()}, progress)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--dataset", choices=("Weather",), required=True)
    parser.add_argument("--family", choices=("P", "B"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--effective-batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--aux-weight", type=float, default=0.1)
    parser.add_argument("--smoke-windows", type=int, default=0,
                        help="explicit development-only subset, never used by the suite manifest")
    args = parser.parse_args()
    if args.effective_batch % args.batch_size:
        parser.error("effective batch must be divisible by microbatch")
    try:
        main(args)
    except Exception:
        path = EXP_ROOT / "results" / SUITE / args.job_id / "failure.json"
        atomic_json_dump({"status": "failed", "traceback": traceback.format_exc(), "timestamp": time.time()}, path)
        raise
