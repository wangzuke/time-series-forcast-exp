#!/usr/bin/env python3
"""Run an official regular forecasting model after mask-aware linear interpolation.

This wrapper owns only the controlled P/B data interface, linear interpolation,
and experiment bookkeeping.  Model classes are imported unmodified from the
authors' repositories.  It deliberately never passes a missingness mask into
PatchTST, DLinear, or iTransformer.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from protocol import EXP_ROOT, ShiftForecastDataset, atomic_json_dump, atomic_torch_save, metrics_sums


BASELINES = EXP_ROOT / "baselines"
MODEL_NAMES = ("PatchTST", "DLinear", "iTransformer")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def gpu_memory_stats(device: torch.device) -> dict[str, float]:
    torch.cuda.synchronize(device)
    mib = 1024 * 1024
    return {
        "peak_gpu_allocated_mib": float(torch.cuda.max_memory_allocated(device) / mib),
        "peak_gpu_reserved_mib": float(torch.cuda.max_memory_reserved(device) / mib),
    }


def git_revision(path: pathlib.Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def source_root(name: str) -> pathlib.Path:
    return {
        "PatchTST": BASELINES / "PatchTST",
        "DLinear": BASELINES / "LTSF-Linear",
        "iTransformer": BASELINES / "iTransformer",
    }[name]


def model_hyperparameters(name: str, dataset: str, channels: int, seq_len: int, pred_len: int) -> dict[str, Any]:
    """Nearest released multivariate pred_len=96 settings, made explicit."""
    if name == "PatchTST":
        large = dataset in {"Weather", "Electricity", "Traffic"}
        return {
            "optimizer": "Adam", "learning_rate": 1e-4, "epochs": 100,
            "patience": {"ETTh1": 100, "Weather": 20, "Electricity": 10, "Traffic": 10}[dataset],
            "batch_size": 24 if dataset == "Traffic" else (32 if dataset == "Electricity" else 128),
            "e_layers": 3, "n_heads": 16 if large else 4,
            "d_model": 128 if large else 16, "d_ff": 256 if large else 128,
            "dropout": 0.2 if large else 0.3, "fc_dropout": 0.2 if large else 0.3,
            "head_dropout": 0.0, "patch_len": 16, "stride": 8,
            "revin": 1, "affine": 0, "subtract_last": 0,
            "schedule": "OneCycleLR(pct_start=0.2) for Electricity/Traffic; released type3 schedule otherwise",
            "script_basis": ("PatchTST_supervised/scripts/PatchTST/traffic.sh" if dataset == "Traffic" else
                             "PatchTST_supervised/scripts/PatchTST/{etth1,weather,electricity}.sh"),
        }
    if name == "DLinear":
        return {
            "optimizer": "Adam", "learning_rate": {"ETTh1": 0.005, "Weather": 1e-4, "Electricity": 1e-3, "Traffic": 0.05}[dataset],
            "epochs": 10, "patience": 3,
            "batch_size": {"ETTh1": 32, "Weather": 16, "Electricity": 16, "Traffic": 16}[dataset],
            "individual": False, "schedule": "released type1 epochwise halving",
            "script_basis": ("LTSF-Linear/scripts/EXP-LongForecasting/Linear/traffic.sh" if dataset == "Traffic" else
                             "LTSF-Linear/scripts/EXP-LongForecasting/Linear/{etth1,weather,electricity}.sh"),
        }
    if name == "iTransformer":
        return {
            "optimizer": "Adam", "learning_rate": {"Electricity": 5e-4, "Traffic": 1e-3}.get(dataset, 1e-4),
            "epochs": 10, "patience": 3,
            "batch_size": 16 if dataset in {"Electricity", "Traffic"} else 32,
            "e_layers": 4 if dataset == "Traffic" else 3, "n_heads": 8, "d_model": 512, "d_ff": 512, "dropout": 0.1,
            "use_norm": True, "schedule": "released type1 epochwise halving",
            "script_basis": ("iTransformer/scripts/multivariate_forecasting/Traffic/iTransformer.sh"
                             if dataset == "Traffic" else
                             "iTransformer/scripts/multivariate_forecasting/ECL/iTransformer.sh"
                             if dataset == "Electricity" else
                             "iTransformer/scripts/multivariate_forecasting/Weather/iTransformer.sh; ETTh1 uses the same multivariate architecture"),
        }
    raise ValueError(name)


def build_model(name: str, dataset: str, channels: int, seq_len: int, pred_len: int, device: torch.device) -> nn.Module:
    hp = model_hyperparameters(name, dataset, channels, seq_len, pred_len)
    if name == "PatchTST":
        repository = source_root(name) / "PatchTST_supervised"
        sys.path.insert(0, str(repository))
        from models.PatchTST import Model as OfficialModel
        config = SimpleNamespace(
            enc_in=channels, seq_len=seq_len, pred_len=pred_len,
            e_layers=hp["e_layers"], n_heads=hp["n_heads"], d_model=hp["d_model"], d_ff=hp["d_ff"],
            dropout=hp["dropout"], fc_dropout=hp["fc_dropout"], head_dropout=hp["head_dropout"],
            individual=0, patch_len=hp["patch_len"], stride=hp["stride"], padding_patch="end",
            revin=hp["revin"], affine=hp["affine"], subtract_last=hp["subtract_last"],
            decomposition=0, kernel_size=25,
        )
    elif name == "DLinear":
        repository = source_root(name)
        sys.path.insert(0, str(repository))
        from models.DLinear import Model as OfficialModel
        config = SimpleNamespace(seq_len=seq_len, pred_len=pred_len, enc_in=channels, individual=False)
    elif name == "iTransformer":
        repository = source_root(name)
        sys.path.insert(0, str(repository))
        from model.iTransformer import Model as OfficialModel
        config = SimpleNamespace(
            seq_len=seq_len, pred_len=pred_len, output_attention=False, use_norm=True,
            d_model=hp["d_model"], n_heads=hp["n_heads"], e_layers=hp["e_layers"], d_ff=hp["d_ff"],
            dropout=hp["dropout"], embed="timeF", freq="h", factor=1, activation="gelu", class_strategy="projection",
        )
    else:
        raise ValueError(name)
    return OfficialModel(config).to(device).float()


def interpolate_history(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-channel linear interpolation inside the known historical window.

    Leading/trailing holes use the nearest available historical observation.
    The protocol guarantees observations in every channel, but the explicit
    check prevents an all-missing channel from silently becoming NaN.
    """
    observed = mask.bool()
    if not bool(observed.any(dim=1).all().item()):
        raise RuntimeError("linear interpolation received an all-missing history channel")
    _, length, _ = x.shape
    positions = torch.arange(length, device=x.device, dtype=torch.long).view(1, length, 1).expand_as(mask)
    prev_marker = torch.where(observed, positions, torch.full_like(positions, -1))
    prev_index = prev_marker.cummax(dim=1).values
    next_marker = torch.where(observed, positions, torch.full_like(positions, length))
    next_index = torch.flip(torch.cummin(torch.flip(next_marker, dims=[1]), dim=1).values, dims=[1])
    prev_value = x.gather(1, prev_index.clamp(0, length - 1))
    next_value = x.gather(1, next_index.clamp(0, length - 1))
    have_prev, have_next = prev_index >= 0, next_index < length
    fraction = (positions - prev_index).to(x.dtype) / (next_index - prev_index).clamp_min(1).to(x.dtype)
    interior = prev_value + (next_value - prev_value) * fraction
    filled = torch.where(have_prev & have_next, interior, torch.where(have_prev, prev_value, next_value))
    return torch.where(observed, x, filled)


def interpolation_audit(batch: dict[str, torch.Tensor]) -> dict[str, Any]:
    filled = interpolate_history(batch["x"], batch["mask"])
    observed = batch["mask"].bool()
    return {
        "method": "per_channel_linear_interpolation_with_nearest_boundary_extension",
        "input_finite": bool(torch.isfinite(batch["x"]).all().item()),
        "output_finite": bool(torch.isfinite(filled).all().item()),
        "observed_values_unchanged": bool(torch.equal(filled[observed], batch["x"][observed])),
        "observed_fraction": float(batch["mask"].mean().item()),
        "history_shape": list(batch["x"].shape),
    }


def prediction(name: str, model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    x = interpolate_history(batch["x"], batch["mask"])
    if name in {"PatchTST", "DLinear"}:
        return model(x)
    return model(x, None, None, None)


def evaluate(model_name: str, model: nn.Module, loader: DataLoader, device: torch.device,
             mean: torch.Tensor, std: torch.Tensor, max_batches: int) -> dict[str, float | int]:
    model.eval()
    objective_sum = 0.0
    windows = 0
    mae_sum = mse_sum = 0.0
    elements = 0
    with torch.no_grad():
        for number, batch in enumerate(loader):
            if max_batches > 0 and number >= max_batches:
                break
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            pred = prediction(model_name, model, batch)
            if pred.shape != batch["y"].shape:
                raise RuntimeError(f"{model_name} returned {tuple(pred.shape)}, expected {tuple(batch['y'].shape)}")
            loss = torch.mean((pred - batch["y"]).square())
            objective_sum += float(loss.item()) * pred.shape[0]
            windows += int(pred.shape[0])
            mae, mse, count = metrics_sums(pred, batch["y_raw"], mean, std)
            mae_sum += mae
            mse_sum += mse
            elements += count
    if not windows or not elements:
        raise RuntimeError("evaluation produced no data")
    return {"objective": objective_sum / windows, "mae_raw": mae_sum / elements,
            "mse_raw": mse_sum / elements, "n_windows": windows, "n_target_elements": elements}


def parser() -> argparse.ArgumentParser:
    parsed = argparse.ArgumentParser(description=__doc__)
    parsed.add_argument("--baseline", choices=MODEL_NAMES, required=True)
    parsed.add_argument("--dataset", choices=("ETTh1", "Weather", "Electricity", "Traffic"), required=True)
    parsed.add_argument("--train-family", choices=("P", "B"), required=True)
    parsed.add_argument("--test-family", choices=("P", "B"), required=True)
    parsed.add_argument("--block-len", type=int, default=24)
    parsed.add_argument("--train-seed", type=int, required=True)
    parsed.add_argument("--test-mask-seed", type=int, required=True)
    parsed.add_argument("--seq-len", type=int, default=96)
    parsed.add_argument("--pred-len", type=int, default=96)
    parsed.add_argument("--missing-rate", type=float, default=0.5)
    parsed.add_argument("--mask-balance", choices=("per_channel", "global"), default="per_channel")
    parsed.add_argument("--num-workers", type=int, default=10,
                        help="DataLoader worker count; 10 matches all three official training entry points.")
    parsed.add_argument("--max-train-batches", type=int, default=-1)
    parsed.add_argument("--max-eval-batches", type=int, default=-1)
    parsed.add_argument("--epochs-override", type=int, default=0,
                        help="Smoke-test only: positive value overrides the released epoch count and is recorded.")
    parsed.add_argument("--patience-override", type=int, default=0,
                        help="Smoke-test only: positive value overrides released patience and is recorded.")
    parsed.add_argument("--checkpoint-path", type=pathlib.Path, required=True)
    parsed.add_argument("--result-path", type=pathlib.Path, required=True)
    parsed.add_argument("--progress-path", type=pathlib.Path, required=True)
    parsed.add_argument("--eval-only", action="store_true")
    return parsed


def main() -> None:
    args = parser().parse_args()
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    set_seed(args.train_seed)
    train_ds = ShiftForecastDataset(args.dataset, "train", args.seq_len, args.pred_len, args.train_family,
                                    args.missing_rate, args.train_seed, args.block_len, dynamic_epoch=True,
                                    mask_balance=args.mask_balance)
    val_ds = ShiftForecastDataset(args.dataset, "val", args.seq_len, args.pred_len, args.train_family,
                                  args.missing_rate, args.train_seed + 10_000, args.block_len, dynamic_epoch=False,
                                  mask_balance=args.mask_balance)
    test_ds = ShiftForecastDataset(args.dataset, "test", args.seq_len, args.pred_len, args.test_family,
                                   args.missing_rate, args.test_mask_seed, args.block_len, dynamic_epoch=False,
                                   mask_balance=args.mask_balance)
    hp = dict(model_hyperparameters(args.baseline, args.dataset, train_ds.n_features, args.seq_len, args.pred_len))
    if args.epochs_override > 0:
        hp["epochs"] = args.epochs_override
    if args.patience_override > 0:
        hp["patience"] = args.patience_override
    batch_size = int(hp["batch_size"])
    loader_options = {"num_workers": args.num_workers, "pin_memory": True}
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **loader_options)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_options)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **loader_options)
    model = build_model(args.baseline, args.dataset, train_ds.n_features, args.seq_len, args.pred_len, device)
    mean, std = torch.as_tensor(train_ds.mean, device=device), torch.as_tensor(train_ds.std, device=device)
    audit_batch = next(iter(DataLoader(train_ds, batch_size=min(batch_size, 8), shuffle=False, **loader_options)))
    audit_batch = {key: value.to(device) for key, value in audit_batch.items()}
    metadata: dict[str, Any] = {
        "baseline": args.baseline, "dataset": args.dataset, "train_family": args.train_family,
        "test_family": args.test_family, "train_seed": args.train_seed, "test_mask_seed": args.test_mask_seed,
        "seq_len": args.seq_len, "pred_len": args.pred_len, "missing_rate": args.missing_rate,
        "block_len": args.block_len, "mask_balance": args.mask_balance, "batch_size": batch_size,
        "data_loader_num_workers": args.num_workers,
        "device": torch.cuda.get_device_name(device),
        "official_repository": str(source_root(args.baseline)), "official_revision": git_revision(source_root(args.baseline)),
        "official_hyperparameters": hp, "preprocessing": interpolation_audit(audit_batch),
        "mask_audit": {"train": train_ds.audit(), "val": val_ds.audit(), "test": test_ds.audit()},
    }
    atomic_json_dump(metadata, args.progress_path.with_name(args.progress_path.stem + "_protocol.json"))
    started = time.time()
    try:
        if args.eval_only:
            model.load_state_dict(torch.load(args.checkpoint_path, map_location=device)["model"])
            result = {**metadata, "mode": "eval_only", "status": "complete", "elapsed_sec": time.time() - started,
                      "test": evaluate(args.baseline, model, test_loader, device, mean, std, args.max_eval_batches),
                      **gpu_memory_stats(device)}
        else:
            optimizer = Adam(model.parameters(), lr=float(hp["learning_rate"]))
            onecycle = OneCycleLR(optimizer, max_lr=float(hp["learning_rate"]), epochs=int(hp["epochs"]),
                                  steps_per_epoch=len(train_loader), pct_start=0.2) if (
                                      args.baseline == "PatchTST" and args.dataset in {"Electricity", "Traffic"}) else None
            best, best_epoch, stale = float("inf"), -1, 0
            for epoch in range(1, int(hp["epochs"]) + 1):
                train_ds.set_epoch(epoch)
                model.train()
                total_loss, seen = 0.0, 0
                for number, batch in enumerate(train_loader):
                    if args.max_train_batches > 0 and number >= args.max_train_batches:
                        break
                    batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                    optimizer.zero_grad(set_to_none=True)
                    pred = prediction(args.baseline, model, batch)
                    loss = torch.mean((pred - batch["y"]).square())
                    if not bool(torch.isfinite(loss).item()):
                        raise RuntimeError(f"non-finite loss at epoch={epoch}, batch={number}")
                    loss.backward()
                    optimizer.step()
                    if onecycle is not None:
                        onecycle.step()
                    total_loss += float(loss.item()) * pred.shape[0]
                    seen += int(pred.shape[0])
                val = evaluate(args.baseline, model, val_loader, device, mean, std, args.max_eval_batches)
                if val["objective"] < best:
                    best, best_epoch, stale = float(val["objective"]), epoch, 0
                    atomic_torch_save({"model": model.state_dict(), "metadata": metadata}, args.checkpoint_path)
                else:
                    stale += 1
                if onecycle is None:
                    # Both released DLinear/iTransformer entries use type1; PatchTST's ETTh1/Weather
                    # scripts use type3 (constant for the first three epochs, then ×0.9 each epoch).
                    if args.baseline in {"DLinear", "iTransformer"}:
                        lr = float(hp["learning_rate"]) * (0.5 ** ((epoch - 1) // 1))
                    elif args.baseline == "PatchTST" and epoch >= 3:
                        lr = float(hp["learning_rate"]) * (0.9 ** (epoch - 3))
                    else:
                        lr = float(hp["learning_rate"])
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                progress = {**metadata, "status": "running", "epoch": epoch, "elapsed_sec": time.time() - started,
                            "train_objective": total_loss / max(1, seen), "train_windows": seen, "val": val,
                            "best_epoch": best_epoch, "best_val_objective": best, "stale_epochs": stale,
                            "learning_rate": optimizer.param_groups[0]["lr"], **gpu_memory_stats(device)}
                atomic_json_dump(progress, args.progress_path)
                print(json.dumps(progress, ensure_ascii=False), flush=True)
                if stale >= int(hp["patience"]):
                    break
            model.load_state_dict(torch.load(args.checkpoint_path, map_location=device)["model"])
            result = {**metadata, "mode": "train", "status": "complete", "elapsed_sec": time.time() - started,
                      "best_epoch": best_epoch, "best_val_objective": best,
                      "test": evaluate(args.baseline, model, test_loader, device, mean, std, args.max_eval_batches),
                      **gpu_memory_stats(device)}
        atomic_json_dump(result, args.result_path)
        atomic_json_dump(result, args.progress_path)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    except Exception as error:
        failed = {**metadata, "status": "failed", "elapsed_sec": time.time() - started,
                  "error": repr(error), "traceback": traceback.format_exc()}
        atomic_json_dump(failed, args.progress_path)
        atomic_json_dump(failed, args.result_path)
        raise


if __name__ == "__main__":
    main()
