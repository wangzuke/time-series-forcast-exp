"""统一训练 / 评估循环。"""
from __future__ import annotations
import os
import time
import json
import random
import argparse
from dataclasses import asdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..data.datasets import MissingForecastDataset, collate
from ..data.timefeatures import time_feature_dim
from ..utils.constants import DATASETS
from .pipelines import build_pipeline, PipelineConfig


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_loader(name, flag, seq_len, pred_len, missing_type, missing_rate, base_seed,
               batch_size, num_workers=2, shuffle=None):
    ds = MissingForecastDataset(
        name=name, flag=flag, seq_len=seq_len, pred_len=pred_len,
        missing_type=missing_type, missing_rate=missing_rate, base_seed=base_seed,
    )
    if shuffle is None:
        shuffle = (flag == "train")
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        drop_last=(flag == "train"), collate_fn=collate, pin_memory=True,
    ), ds


def epoch_loop(model, loader, optimizer, device, criterion, train: bool, aux_weight: float = 1.0):
    model.train(train)
    total = 0.0
    total_n = 0
    aux_total = 0.0
    mse_total = 0.0
    mae_total = 0.0
    impute_mae_total = 0.0
    impute_n = 0
    t_start = time.time()
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.set_grad_enabled(train):
            out = model(batch)
            y = batch["y"]
            pred = out["forecast"]
            loss = criterion(pred, y)
            aux = out.get("aux_loss", torch.tensor(0.0, device=device))
            full = loss + aux_weight * aux
            if train:
                optimizer.zero_grad()
                full.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        bs = y.size(0)
        total += loss.item() * bs
        aux_total += float(aux.detach().item()) * bs
        total_n += bs
        with torch.no_grad():
            mse_total += ((pred - y) ** 2).mean(dim=(1, 2)).sum().item()
            mae_total += (pred - y).abs().mean(dim=(1, 2)).sum().item()
            if "impute" in out:
                x_raw = batch["x_raw"]
                mask = batch["mask"]
                imp = out["impute"]
                # 仅评估缺失位置（mask=0）的补值误差
                miss = (1.0 - mask)
                if miss.sum() > 0:
                    err = ((imp - x_raw) ** 2 * miss).sum() / miss.sum()
                    impute_mae_total += float(err.item()) * bs
                    impute_n += bs
    dt = time.time() - t_start
    metrics = {
        "loss": total / max(1, total_n),
        "aux_loss": aux_total / max(1, total_n),
        "mse": mse_total / max(1, total_n),
        "mae": mae_total / max(1, total_n),
        "impute_mse": impute_mae_total / impute_n if impute_n > 0 else 0.0,
        "time_sec": dt,
        "n_samples": total_n,
    }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", default="baseline",
                        choices=["baseline", "simple", "saits", "misstsm", "crib", "coifnet"])
    parser.add_argument("--predictor", default="iTransformer",
                        choices=["DLinear", "PatchTST", "iTransformer"])
    parser.add_argument("--impute", default="zero",
                        choices=["zero", "mean", "forward", "linear", "none"])
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--missing_type", default="none",
                        choices=["none", "random_point", "continuous_segment", "variable_channel", "mixed"])
    parser.add_argument("--missing_rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", default="results")
    parser.add_argument("--tag", default="")
    parser.add_argument("--aux_weight", type=float, default=1.0)
    # SAITS 预训练
    parser.add_argument("--saits_pretrain_epochs", type=int, default=0,
                        help="如果 method=saits 且 >0，则先单独预训练 SAITS")
    # 用更小数据集做快速烟测
    parser.add_argument("--max_train_batches", type=int, default=-1)
    args = parser.parse_args()

    set_seed(args.seed)

    meta = DATASETS[args.dataset]
    n_channels = meta["n_features"]
    freq = meta["freq"]
    t_feat_dim = time_feature_dim(freq)

    cfg = PipelineConfig(
        method=args.method,
        predictor=args.predictor,
        impute_strategy=args.impute,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        n_channels=n_channels,
        time_feat_dim=t_feat_dim,
    )

    train_loader, _ = get_loader(args.dataset, "train", args.seq_len, args.pred_len,
                                 args.missing_type, args.missing_rate, args.seed,
                                 args.batch_size, args.num_workers)
    val_loader, _ = get_loader(args.dataset, "val", args.seq_len, args.pred_len,
                               args.missing_type, args.missing_rate, args.seed,
                               args.batch_size, args.num_workers)
    test_loader, _ = get_loader(args.dataset, "test", args.seq_len, args.pred_len,
                                args.missing_type, args.missing_rate, args.seed,
                                args.batch_size, args.num_workers)

    model = build_pipeline(cfg).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())

    # SAITS 预训练（可选）
    if args.method == "saits" and args.saits_pretrain_epochs > 0:
        from ..imputation.saits import random_mit_mask, masked_mae
        opt_s = torch.optim.AdamW(model.saits.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for ep in range(args.saits_pretrain_epochs):
            model.train()
            total = 0.0; n = 0
            for batch in train_loader:
                batch = {k: v.to(args.device, non_blocking=True) for k, v in batch.items()}
                ind, new_mask = random_mit_mask(batch["mask"], p=cfg.saits_mit_rate)
                X_obs = batch["x_obs"] * (new_mask / batch["mask"].clamp(min=1e-8))
                loss, _, _, _ = model.saits.compute_loss(X_obs, new_mask, batch["x_raw"], ind)
                opt_s.zero_grad(); loss.backward(); opt_s.step()
                total += float(loss.item()) * batch["x_raw"].size(0); n += batch["x_raw"].size(0)
            print(f"[SAITS pretrain ep{ep+1}/{args.saits_pretrain_epochs}] loss={total/max(1,n):.4f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    patience = args.patience
    bad = 0
    history = []
    for ep in range(args.epochs):
        if hasattr(train_loader.dataset, "set_epoch_seed_offset"):
            train_loader.dataset.set_epoch_seed_offset(ep)
        tr = epoch_loop(model, train_loader, optimizer, args.device, criterion, train=True,
                        aux_weight=args.aux_weight)
        with torch.no_grad():
            va = epoch_loop(model, val_loader, None, args.device, criterion, train=False)
        history.append({"epoch": ep + 1, "train": tr, "val": va})
        print(f"[ep{ep+1}/{args.epochs}] tr_mse={tr['mse']:.4f} tr_mae={tr['mae']:.4f} | "
              f"va_mse={va['mse']:.4f} va_mae={va['mae']:.4f} | tr_time={tr['time_sec']:.1f}s")
        if va["mse"] < best_val - 1e-6:
            best_val = va["mse"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"early stop at epoch {ep+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    with torch.no_grad():
        te = epoch_loop(model, test_loader, None, args.device, criterion, train=False)

    peak_mem = torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0.0

    out = {
        "config": vars(args),
        "n_channels": n_channels,
        "n_params": n_params,
        "best_val_mse": best_val,
        "test": te,
        "peak_mem_mb": peak_mem,
        "history": history,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    tag = args.tag or (
        f"{args.dataset}_{args.method}_{args.predictor}_{args.impute}_"
        f"{args.missing_type}_{int(args.missing_rate*100)}_h{args.pred_len}_s{args.seed}"
    )
    out_path = os.path.join(args.out_dir, f"{tag}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print("saved", out_path)


if __name__ == "__main__":
    main()
