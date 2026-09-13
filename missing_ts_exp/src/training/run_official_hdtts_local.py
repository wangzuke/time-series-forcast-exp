"""Run official HD-TTS modules on local spatial datasets.

Use the ``hd-tts`` conda environment:
    MPLCONFIGDIR=/tmp/mpl-cache /media/data1/wangzk/miniconda3/envs/hd-tts/bin/python \
        -m missing_ts_exp.src.training.run_official_hdtts_local --model mrmsgf
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import subprocess
import sys
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.utils import dense_to_sparse

from .run_hdtts_reliable_spatial import load_spatial_dataset, make_loaders
from ..models.hdtts_official_reliable import (
    OFFICIAL_HDTTS,
    OfficialMRMSGFHDTTSModel,
    graph_region_assignment,
)
from ..data.hdtts_missing import sha256_file


if str(OFFICIAL_HDTTS) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_HDTTS))

from lib.nn.models.hdtts_model import HDTTSModel  # noqa: E402


ROOT = pathlib.Path(__file__).resolve().parents[3]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def source_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def official_hdtts_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=OFFICIAL_HDTTS, text=True).strip()
    except Exception:
        return "unknown"


def masked_mse(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return ((pred - y).square() * mask).sum() / mask.sum().clamp_min(1.0)


def masked_mae(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return ((pred - y).abs() * mask).sum() / mask.sum().clamp_min(1.0)


def limit_batches(loader, max_batches: int):
    for idx, batch in enumerate(loader):
        if max_batches >= 0 and idx >= max_batches:
            break
        yield batch


def parse_mp_method(value: str) -> str | list[str]:
    methods = [item.strip() for item in value.split(",") if item.strip()]
    valid = {"propconv", "diffconv", "diffconv_sym", "anisoconv"}
    if not methods or any(item not in valid for item in methods):
        raise ValueError(f"Invalid --mp_method {value!r}; choices are {sorted(valid)}")
    return methods[0] if len(methods) == 1 else methods


def load_region_assignment(path: str, adjacency: torch.Tensor) -> torch.Tensor:
    if not path:
        raise ValueError("MR-MSGF requires --region_assignment_path generated from the physical graph.")
    assignment = torch.from_numpy(np.load(path).astype(np.float32))
    if assignment.ndim != 2 or assignment.size(0) != adjacency.size(0):
        raise ValueError(f"region assignment shape {tuple(assignment.shape)} is incompatible with {tuple(adjacency.shape)}")
    if not torch.allclose(assignment.sum(dim=1), torch.ones(assignment.size(0))):
        raise ValueError("region assignment must assign every node to exactly one region")
    return assignment


def build_model(args: argparse.Namespace, adjacency: torch.Tensor, region_assignment: torch.Tensor | None = None) -> nn.Module:
    n_nodes = int(adjacency.size(0))
    cls = HDTTSModel if args.model == "official_base" else OfficialMRMSGFHDTTSModel
    kwargs = dict(
        input_size=1,
        hidden_size=args.hidden,
        n_nodes=n_nodes,
        horizon=args.pred_len,
        rnn_layers=args.rnn_layers,
        pooling_layers=args.pooling_layers,
        mask_size=1,
        dilation=args.dilation,
        mp_method=parse_mp_method(args.mp_method),
        mp_kernel_size=1,
        recursive_lifting=True,
        fully_connected_readout=args.fully_connected_readout,
        multi_step_scores=False,
        activation="elu",
        dropout=args.dropout,
    )
    if cls is OfficialMRMSGFHDTTSModel:
        kwargs.update(
            physical_adjacency=adjacency,
            region_assignment=region_assignment if region_assignment is not None else graph_region_assignment(adjacency, args.max_region_size),
            variant=args.mrmsgf_variant,
            graph_topk=args.graph_topk,
            reliability_beta=args.reliability_beta,
            disable_overlap_reliability=args.disable_overlap_reliability,
            disable_readout_reliability=args.disable_readout_reliability,
        )
    return cls(**kwargs)


def epoch_loop(
    model: nn.Module,
    loader,
    optimizer: Optional[torch.optim.Optimizer],
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    device: torch.device,
    train: bool,
    max_batches: int,
    train_loss: str,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> dict[str, float]:
    model.train(train)
    total_loss = 0.0
    total_normalized_mae = 0.0
    total_abs_error = 0.0
    total_sq_error = 0.0
    total_abs_pct_error = 0.0
    total_values = 0.0
    total_batches = 0
    diagnostics: dict[str, float] = {}
    started = time.time()
    for batch in limit_batches(loader, max_batches):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        x = batch["x"].unsqueeze(-1)
        mask = batch["mask"].unsqueeze(-1)
        y = batch["y"].unsqueeze(-1)
        y_mask = batch["y_mask"].unsqueeze(-1)
        with torch.set_grad_enabled(train):
            y_hat, *_ = model(x, edge_index, edge_weight=edge_weight, input_mask=mask)
            loss = masked_mae(y_hat, y, y_mask) if train_loss == "mae" else masked_mse(y_hat, y, y_mask)
            if train:
                assert optimizer is not None
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        scale = std.view(1, 1, -1, 1)
        offset = mean.view(1, 1, -1, 1)
        raw_error = y_hat.detach() * scale + offset - (y * scale + offset)
        valid = y_mask.float()
        denom = float(valid.sum().item())
        total_loss += float(loss.detach().item())
        total_normalized_mae += float(masked_mae(y_hat.detach(), y, y_mask).item())
        total_abs_error += float((raw_error.abs() * valid).sum().item())
        total_sq_error += float((raw_error.square() * valid).sum().item())
        total_abs_pct_error += float((raw_error.abs() / (y * scale + offset).abs().clamp_min(1e-3) * valid).sum().item())
        total_values += denom
        total_batches += 1
        if hasattr(model, "diagnostic_summary"):
            for key, value in model.diagnostic_summary().items():
                diagnostics[key] = diagnostics.get(key, 0.0) + value
    return {
        "loss": total_loss / max(1, total_batches),
        "normalized_mae": total_normalized_mae / max(1, total_batches),
        "mae": total_abs_error / max(1.0, total_values),
        "rmse": math.sqrt(total_sq_error / max(1.0, total_values)),
        "mape": 100.0 * total_abs_pct_error / max(1.0, total_values),
        "n_values": total_values,
        "time_sec": time.time() - started,
        **{f"diagnostic_{key}": value / max(1, total_batches) for key, value in diagnostics.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mrmsgf", choices=["official_base", "mrmsgf"])
    parser.add_argument("--dataset", default="METR-LA", choices=["METR-LA", "AQI36"])
    parser.add_argument("--seq_len", type=int, default=24)
    parser.add_argument("--pred_len", type=int, default=12)
    parser.add_argument("--missing_type", default="variable_channel",
                        choices=["none", "random_point", "continuous_segment", "variable_channel", "mixed", "point", "block_t"])
    parser.add_argument("--missing_rate", type=float, default=0.5)
    parser.add_argument("--mask_path", default="", help="Path to a fixed observed mask .npy with 1=observed, 0=missing.")
    parser.add_argument("--p_noise", type=float, default=0.05)
    parser.add_argument("--p_fault", type=float, default=0.01)
    parser.add_argument("--min_seq", type=int, default=12)
    parser.add_argument("--max_seq", type=int, default=48)
    parser.add_argument("--calibrate_block_t", action="store_true")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--rnn_layers", type=int, default=3)
    parser.add_argument("--pooling_layers", type=int, default=3)
    parser.add_argument("--dilation", type=int, default=4)
    parser.add_argument("--mp_method", default="anisoconv,anisoconv",
                        help="One method or a comma-separated pair, e.g. anisoconv,anisoconv.")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--fully_connected_readout", action="store_true")
    parser.add_argument("--mrmsgf_variant", default="full",
                        choices=sorted(OfficialMRMSGFHDTTSModel.VALID_VARIANTS))
    parser.add_argument("--max_region_size", type=int, default=8)
    parser.add_argument("--region_assignment_path", default="",
                        help="Verified fixed node-to-region assignment (.npy); required for --model mrmsgf.")
    parser.add_argument("--graph_topk", type=int, default=8)
    parser.add_argument("--reliability_beta", type=float, default=1.0)
    parser.add_argument("--disable_overlap_reliability", action="store_true")
    parser.add_argument("--disable_readout_reliability", action="store_true")
    parser.add_argument("--train_loss", default="mae", choices=["mae", "mse"])
    parser.add_argument("--max_train_batches", type=int, default=-1)
    parser.add_argument("--max_eval_batches", type=int, default=-1)
    parser.add_argument("--out_dir", default=str(ROOT / "missing_ts_exp" / "results" / "r0902_official_hdtts_local"))
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    set_seed(args.seed)
    values, valid_mask, adj, node_ids, data_path = load_spatial_dataset(args.dataset)
    loaders, mean, std, starts = make_loaders(args, values, valid_mask)
    adjacency = torch.from_numpy(adj).float()
    edge_index, edge_weight = dense_to_sparse(adjacency)
    device = torch.device(args.device)
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    region_assignment = load_region_assignment(args.region_assignment_path, adjacency) if args.model == "mrmsgf" else None
    model = build_model(args, adjacency, region_assignment).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = math.inf
    best_state = None
    bad = 0
    history = []
    for epoch in range(args.epochs):
        train_metrics = epoch_loop(model, loaders["train"], optimizer, edge_index, edge_weight, device, True, args.max_train_batches, args.train_loss, torch.from_numpy(mean).to(device), torch.from_numpy(std).to(device))
        with torch.no_grad():
            val_metrics = epoch_loop(model, loaders["val"], None, edge_index, edge_weight, device, False, args.max_eval_batches, args.train_loss, torch.from_numpy(mean).to(device), torch.from_numpy(std).to(device))
        history.append({"epoch": epoch + 1, "train": train_metrics, "val": val_metrics})
        print(
            f"[{args.model} ep{epoch + 1}/{args.epochs}] "
            f"train_loss={train_metrics['loss']:.4f} train_mae={train_metrics['mae']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_mae={val_metrics['mae']:.4f}"
        )
        if val_metrics["loss"] < best_val - 1e-8:
            best_val = val_metrics["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("No AQI36 checkpoint was selected.")
    model.load_state_dict(best_state)
    with torch.no_grad():
        test_metrics = epoch_loop(model, loaders["test"], None, edge_index, edge_weight, device, False, args.max_eval_batches, args.train_loss, torch.from_numpy(mean).to(device), torch.from_numpy(std).to(device))

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or (
        f"{args.model}_{args.dataset}_{args.missing_type}_{int(args.missing_rate * 100)}_"
        f"h{args.pred_len}_seed{args.seed}"
    )
    checkpoint_path = out_dir / tag / "best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "best_val_loss": best_val, "config": vars(args)}, checkpoint_path)

    out = {
        "model": args.model,
        "source_commit": source_commit(),
        "official_hdtts_commit": official_hdtts_commit(),
        "official_hdtts_repo": str(OFFICIAL_HDTTS.relative_to(ROOT)),
        "implementation": "official HD-TTS backbone + explicit MR-MSGF modules" if args.model == "mrmsgf" else "unmodified official HD-TTS model class",
        "config": vars(args),
        "data_path": data_path,
        "data_sha256": sha256_file(data_path),
        "mask_sha256": sha256_file(args.mask_path) if args.mask_path else None,
        "n_nodes": int(values.shape[1]),
        "node_ids_head": node_ids[:10],
        "adjacency_shape": list(adj.shape),
        "region_assignment_path": args.region_assignment_path if args.model == "mrmsgf" else None,
        "region_assignment_sha256": sha256_file(args.region_assignment_path) if args.model == "mrmsgf" else None,
        "split_sizes": {k: int(len(v)) for k, v in starts.items()},
        "valid_value_rate": float(valid_mask.mean()),
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "best_val_loss": float(best_val),
        "best_checkpoint": str(checkpoint_path),
        "test": test_metrics,
        "history": history,
    }
    out_path = out_dir / f"{tag}.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print("saved", out_path)


if __name__ == "__main__":
    main()
