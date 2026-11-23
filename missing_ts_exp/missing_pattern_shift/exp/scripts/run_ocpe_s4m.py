#!/usr/bin/env python3
"""Resumable paired S4M training, including its non-module prototype memory."""
from __future__ import annotations

import argparse
import fcntl
import os
import time
import traceback
import hashlib
import torch
from torch.utils.data import DataLoader
from protocol import EXP_ROOT, atomic_json_dump, atomic_torch_save, stable_seed
from ocpe_backbone_common import (SUITE, OCPES4M, FusionDataset, seed_all, digest,
    source_fingerprints, initialize_memory, make_s4m_optimizers, mask_fingerprint,
    mask_audit, memory_stats, runtime_metadata, move_batch, rng_state, restore_rng)
from run_ocpe_fusion_s4 import evaluate, sequence_hash


def fixed_evaluate(model, loader, device, seed, output=None):
    saved = rng_state()
    try:
        seed_all(seed)
        return evaluate(model, loader, device, output)
    finally:
        restore_rng(saved)


def train_epoch(model, loader, optimizers, args, device, progress, epoch):
    model.train()
    sums = torch.zeros(3, dtype=torch.float64, device=device)
    hasher = hashlib.sha256()
    samples, batches = 0, 0
    started, last_report = time.perf_counter(), 0
    for number, batch in enumerate(loader):
        sequence_hash(hasher, batch)
        batch = move_batch(batch, device)
        for optimizer in optimizers.values(): optimizer.zero_grad(set_to_none=True)
        pred, nll = model(batch["x"], batch["mask"])
        valid = batch["target_valid"]
        error = torch.where(valid.bool(), pred - batch["y"], 0)
        sse, count = error.square().sum(), valid.sum()
        if count <= 0: raise ValueError("No valid forecast targets")
        loss = sse / count + args.aux_weight * nll.mean()
        if not torch.isfinite(loss): raise FloatingPointError("Non-finite training objective")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.backbone.parameters(), float("inf"), error_if_nonfinite=True)
        if model.enabled:
            torch.nn.utils.clip_grad_norm_(model.ocpe_parameters(), 1, error_if_nonfinite=True)
        for optimizer in optimizers.values(): optimizer.step()
        sums += torch.stack((sse.detach(), count, nll.detach().sum())).double()
        samples += len(pred); batches += 1
        if time.perf_counter() - last_report >= 10:
            atomic_json_dump({"status": "training", "job_id": args.job_id, "epoch": epoch,
                              "batch": number + 1, "total_batches": len(loader), "pid": os.getpid(),
                              "updated_at": time.time(), **memory_stats()}, progress)
            last_report = time.perf_counter()
    if not samples: raise RuntimeError("No training windows")
    values = sums.cpu().tolist()
    return {"mse": values[0] / values[1], "valid_targets": int(values[1]),
            "observation_nll": values[2] / samples, "samples": samples, "batches": batches,
            "seconds": time.perf_counter() - started,
            "effective_mask_sequence_sha256": hasher.hexdigest()}


def main(args):
    torch.set_num_threads(args.threads); seed_all(args.seed)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    device = torch.device("cuda:0"); torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats()
    root = EXP_ROOT / "results" / args.suite / args.job_id
    checkpoints = EXP_ROOT / "checkpoints" / args.suite
    checkpoints.mkdir(parents=True, exist_ok=True)
    train = FusionDataset(args.dataset, "train", args.family, args.seed, True)
    val = FusionDataset(args.dataset, "val", args.family, args.seed + 10000)
    if args.smoke_windows:
        train.n_samples = min(len(train), args.smoke_windows); val.n_samples = min(len(val), args.smoke_windows)
    model = OCPES4M(args.dataset, train.n_features, args.variant == "OCPE-S4M", args.seed, device)
    runtime = runtime_metadata()
    if not runtime["has_cuda_extension"]: raise RuntimeError("Compiled S4 CUDA kernel required")
    record = {"suite": args.suite, "job_id": args.job_id, "variant": args.variant,
              "dataset": args.dataset, "train_family": args.family, "train_seed": args.seed,
              "protocol": "ocpe_fusion_valid_weather_B_uniform_sep24_v2",
              "source": source_fingerprints(), "data_quality": train.quality,
              "model": {"s4m_layers": 4, "s4m_width": model.backbone.d_model, "ocpe": model.cfg.to_dict()},
              "batch_size": args.batch_size, "effective_batch": args.batch_size,
              "workers": args.workers, "threads": args.threads, "epochs": args.epochs,
              "patience": args.patience, "aux_weight": args.aux_weight, "smoke_windows": args.smoke_windows,
              "parameter_counts": model.parameter_counts(), "optimizer": "backbone official SGD; OCPE AdamW 1e-3",
              "gradient_clipping": "backbone none (finite check); OCPE norm1",
              "train_mask_epoch1": mask_fingerprint(train, 1), "train_mask_epoch2": mask_fingerprint(train, 2),
              "validation_mask": mask_fingerprint(val), "validation_effective_mask": mask_fingerprint(val, effective=True),
              "train_mask_audit": mask_audit(train), "validation_mask_audit": mask_audit(val)}
    fingerprint = digest(record); record.update(fingerprint=fingerprint, runtime=runtime)
    protocol = root / "protocol.json"
    if protocol.exists():
        import json
        if json.loads(protocol.read_text())["fingerprint"] != fingerprint:
            raise RuntimeError("Frozen job identity changed")
    atomic_json_dump(record, protocol)
    optimizers = make_s4m_optimizers(model)
    options = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, persistent_workers=False)
    if args.workers: options["prefetch_factor"] = 2
    def loader(ds, order=None, epoch=0):
        return DataLoader(ds, sampler=order, shuffle=False,
                          generator=torch.Generator().manual_seed(args.seed + 500000 + epoch), **options)
    best_path, last_path = (checkpoints / f"{args.job_id}.{suffix}.pt" for suffix in ("best", "last"))
    best, stale, start_epoch, history = float("inf"), 0, 1, []
    complete_path = root / "training_complete.json"
    if not complete_path.exists():
        if last_path.exists():
            state = torch.load(last_path, map_location=device, weights_only=False)
            if state["fingerprint"] != fingerprint: raise RuntimeError("Resume identity mismatch")
            model.load_state_dict(state["model"]); model.load_memory(state["memory"])
            for key, optimizer in optimizers.items(): optimizer.load_state_dict(state["optimizers"][key])
            best, stale, start_epoch, history = state["best"], state["stale"], state["epoch"] + 1, state["history"]
            warm = state["warm"]; restore_rng(state["rng"])
        else:
            warm = initialize_memory(model, train, args.seed, args.suite)
        started = time.time()
        for epoch in range(start_epoch, args.epochs + 1):
            if stale >= args.patience: break
            train.set_epoch(epoch)
            order = torch.randperm(len(train), generator=torch.Generator().manual_seed(args.seed + 100000 + epoch)).tolist()
            stats = train_epoch(model, loader(train, order, epoch), optimizers, args, device, root / "progress.json", epoch)
            value = fixed_evaluate(model, loader(val), device,
                stable_seed(SUITE, args.dataset, "val", args.family, args.seed))["mse_standardized"]
            improved = value < best
            if improved: best, stale = value, 0
            else: stale += 1
            history.append({"epoch": epoch, "train": stats, "val_mse": value, "improved": improved})
            state = {"model": model.state_dict(), "memory": model.memory_state(cpu=True),
                     "epoch": epoch, "fingerprint": fingerprint, "warm": warm}
            if improved: atomic_torch_save({**state, "val_mse": value}, best_path)
            atomic_torch_save({**state, "optimizers": {k: o.state_dict() for k, o in optimizers.items()},
                              "best": best, "stale": stale, "history": history, "rng": rng_state()}, last_path)
            import json
            print(json.dumps(history[-1]), flush=True)
        atomic_json_dump({"status": "complete", "fingerprint": fingerprint, "job_id": args.job_id,
                          "best_val_mse": best, "history": history, "warm": warm,
                          "training_seconds_this_session": time.time() - started,
                          "stop_reason": "early_stopping" if stale >= args.patience else "max_epochs",
                          **memory_stats()}, complete_path)
    else:
        import json
        if json.loads(complete_path.read_text())["fingerprint"] != fingerprint:
            raise RuntimeError("Completed training identity mismatch")
    state = torch.load(best_path, map_location=device, weights_only=False)
    if state["fingerprint"] != fingerprint: raise RuntimeError("Best checkpoint identity mismatch")
    model.load_state_dict(state["model"]); model.load_memory(state["memory"])
    for family in ("P", "Bu"):
        for seed in (9101, 9102):
            output = root / f"test_{family}_m{seed}.json"
            if output.exists():
                import json
                old = json.loads(output.read_text())
                if old.get("fingerprint") == fingerprint and old.get("status") == "complete": continue
                raise RuntimeError("Existing evaluation incompatible")
            ds = FusionDataset(args.dataset, "test", family, seed)
            if args.smoke_windows: ds.n_samples = min(len(ds), args.smoke_windows)
            atomic_json_dump({"status": "evaluating", "job_id": args.job_id, "family": family,
                              "mask_seed": seed, "updated_at": time.time()}, root / "progress.json")
            result = fixed_evaluate(model, loader(ds), device,
                stable_seed(SUITE, args.dataset, "test", family, seed, args.seed), output)
            result.update(status="complete", fingerprint=fingerprint, job_id=args.job_id,
                          test_family=family, test_mask_seed=seed, mask_fingerprint=mask_fingerprint(ds),
                          mask_audit=mask_audit(ds), best_epoch=state["epoch"])
            atomic_json_dump(result, output)
            print(f"test {family} m{seed}: {result['mse_standardized']}", flush=True)
    atomic_json_dump({"status": "complete", "job_id": args.job_id, "fingerprint": fingerprint,
                      "best_epoch": state["epoch"], "updated_at": time.time(), **memory_stats()}, root / "progress.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("S4M", "OCPE-S4M"), required=True)
    parser.add_argument("--dataset", choices=("ETTh1", "Weather", "Electricity"), required=True)
    parser.add_argument("--family", choices=("P", "Bu"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--suite", default=SUITE)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--aux-weight", type=float, default=.1)
    parser.add_argument("--smoke-windows", type=int, default=0)
    args = parser.parse_args()
    root = EXP_ROOT / "results" / args.suite / args.job_id
    root.mkdir(parents=True, exist_ok=True)
    try:
        with (root / "job.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            main(args)
    except Exception:
        atomic_json_dump({"status": "failed", "traceback": traceback.format_exc(),
                          "timestamp": time.time()}, root / "failure.json")
        raise
