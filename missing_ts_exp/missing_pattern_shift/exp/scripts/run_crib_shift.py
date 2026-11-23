"""Thin experiment adapter for the authors' CRIB implementation.

The model and its loss terms are imported from the pinned author repository.
This file only supplies the project's window/mask protocol, run isolation,
checkpointing, and paired cross-pattern evaluation.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

HERE = Path(__file__).resolve()
EXP = HERE.parents[1]
REPO = EXP / "external_repos" / "CRIB"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE.parent))

# CRIB's eager package __init__ files create a circular import in the pinned
# upstream commit. Registering package search paths avoids running those eager
# initializers without changing any upstream file.
for package_name, package_dir in (("TSL_models", "TSL_models"), ("utils", "utils")):
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(REPO / package_dir)]
        sys.modules[package_name] = package

from TSL_models.CRIB import CRIB  # noqa: E402
from utils.args import parse_args as parse_crib_args  # noqa: E402
from protocol import ShiftForecastDataset, metrics_sums  # noqa: E402


class CribWindowDataset(Dataset):
    def __init__(self, source: ShiftForecastDataset, patch_len: int):
        self.source = source
        self.patch_len = patch_len

    def __len__(self):
        return self.source.n_samples

    def set_epoch(self, epoch: int):
        self.source.set_epoch(epoch)

    def __getitem__(self, index: int):
        ds = self.source
        x = ds.x[index:index + ds.seq_len]
        y0 = index + ds.seq_len
        y = ds.x[y0:y0 + ds.pred_len]
        mask = ds.mask_for(index)
        # Author model expects [batch, patches, variables, patch length].
        x = torch.from_numpy(x.copy()).reshape(ds.seq_len // self.patch_len, self.patch_len, ds.n_features).permute(0, 2, 1)
        mask = torch.from_numpy(mask.copy()).reshape(ds.seq_len // self.patch_len, self.patch_len, ds.n_features).permute(0, 2, 1)
        return x, torch.from_numpy(y.copy()), mask, torch.from_numpy(ds.y_raw[y0:y0 + ds.pred_len].copy())


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device, mean, std, use_amp):
    model.eval()
    mae_sum = mse_sum = 0.0
    count = 0
    with torch.no_grad():
        for x, y, mask, y_raw in loader:
            x, y, mask = x.to(device).float(), y.to(device).float(), mask.to(device).float()
            x = x * mask
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(x, x_mark=None, test_flag=True)
                pred = out[4]
            ma, ms, n = metrics_sums(pred.float(), y_raw.to(device), mean.to(device), std.to(device))
            mae_sum += ma
            mse_sum += ms
            count += n
    model.train()
    return {"mae": mae_sum / count, "mse": mse_sum / count, "n": count}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["ETTh1", "Weather", "Electricity"])
    p.add_argument("--train-family", required=True, choices=["P", "B"])
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--mask-seeds", default="9101,9102")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--patch-len", type=int, default=8)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--max-train-samples", type=int, default=0, help="Smoke test only; 0 uses all windows")
    p.add_argument("--max-test-samples", type=int, default=0, help="Smoke test only; 0 uses all windows")
    p.add_argument("--use-amp", action="store_true")
    p.add_argument("--tag", default="s4m_robustness_1114")
    args = p.parse_args()
    seed_all(args.seed)
    if 96 % args.patch_len:
        raise ValueError("patch length must divide 96")

    crib_args = parse_crib_args([
        "--dataset", {"Electricity": "Elec"}.get(args.dataset, args.dataset),
        "--model", "CRIB", "--seq_len", "96", "--pred_len", "96",
        "--patch_len", str(args.patch_len), "--batch_size", str(args.batch_size),
        "--model_dim", "32", "--enc_num", "3", "--dec_num", "1",
        "--learning_rate", "0.001", "--num_workers", str(args.workers),
    ])
    crib_args.use_amp = args.use_amp
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    crib_args.device = device

    train_src = ShiftForecastDataset(args.dataset, "train", 96, 96, args.train_family, .5, args.seed, 24, True, "per_channel")
    val_src = ShiftForecastDataset(args.dataset, "val", 96, 96, args.train_family, .5, args.seed + 7000, 24, False, "per_channel")
    train_ds, val_ds = CribWindowDataset(train_src, args.patch_len), CribWindowDataset(val_src, args.patch_len)
    if args.max_train_samples:
        train_ds = torch.utils.data.Subset(train_ds, range(min(args.max_train_samples, len(train_ds))))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)

    model = CRIB(crib_args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    mae = nn.L1Loss()
    mse = nn.MSELoss()
    run_id = f"CRIB_{args.dataset}_H96_{args.train_family}_b24_s{args.seed}"
    out_dir = EXP / "results" / args.tag / run_id
    ckpt_dir = EXP / "checkpoints" / args.tag
    log_dir = EXP / "logs" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{run_id}.pt"
    log_path = log_dir / f"{run_id}.jsonl"
    mean = torch.from_numpy(train_src.mean)
    std = torch.from_numpy(train_src.std)
    epoch_logs = []
    start = time.time()
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

    for epoch in range(args.epochs):
        train_src.set_epoch(epoch)
        model.train()
        losses = []
        for x, y, mask, _ in train_loader:
            x, y, mask = x.to(device).float(), y.to(device).float(), mask.to(device).float()
            x = x * mask
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.use_amp):
                output = model(x, x_mark=None, test_flag=False)
                prediction_loss = mae(output[4], y)
                consistency_loss = mse(output[0], output[2])
                loss = prediction_loss + consistency_loss + 1e-6 * output[6]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        epoch_info = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "elapsed_seconds": time.time() - start}
        epoch_logs.append(epoch_info)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(epoch_info) + "\n")
        torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch + 1, "args": vars(args)}, ckpt_path)
        print(json.dumps(epoch_info), flush=True)

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_results = {}
    for family in ("P", "B"):
        for mask_seed in map(int, args.mask_seeds.split(",")):
            src = ShiftForecastDataset(args.dataset, "test", 96, 96, family, .5, mask_seed, 24, False, "per_channel")
            eval_ds = CribWindowDataset(src, args.patch_len)
            if args.max_test_samples:
                eval_ds = torch.utils.data.Subset(eval_ds, range(min(args.max_test_samples, len(eval_ds))))
            loader = DataLoader(eval_ds, batch_size=max(args.batch_size, 8), shuffle=False, num_workers=args.workers)
            test_results[f"{args.train_family}_to_{family}_mask{mask_seed}"] = evaluate(model, loader, device, mean, std, args.use_amp)
    result = {
        "model": "CRIB", "dataset": args.dataset, "train_family": args.train_family,
        "train_seed": args.seed, "test_mask_seeds": list(map(int, args.mask_seeds.split(","))),
        "protocol": {"seq_len": 96, "pred_len": 96, "missing_rate": .5, "block_len": 24, "mask_balance": "per_channel"},
        "source_commit": "a457672c7b0152f74c9c2d8dba2a9c886405519",
        "author_hyperparameters": {"epochs": args.epochs, "learning_rate": .001, "batch_size": args.batch_size, "patch_len": args.patch_len, "model_dim": 32, "encoder_layers": 3, "loss": "MAE + consistency MSE + 1e-6*KL", "amp": args.use_amp},
        "selected_epoch": checkpoint["epoch"], "selection": "last epoch, matching the authors' released training loop",
        "tests": test_results, "elapsed_seconds": time.time() - start,
        "peak_gpu_memory_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 2) if torch.cuda.is_available() else 0,
        "checkpoint": str(ckpt_path), "epoch_logs": epoch_logs,
    }
    result_path = out_dir / "result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"finished": run_id, "result": str(result_path), "test_keys": list(test_results)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
