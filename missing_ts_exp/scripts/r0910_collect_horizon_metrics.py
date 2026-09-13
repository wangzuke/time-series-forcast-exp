#!/usr/bin/env python3
"""Collect unmasked future MAE/RMSE at every forecast horizon from a checkpoint."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from missing_ts_exp.src.training.run_mrmsgf_official_controlled import build_predictor, make_datamodule


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_json", required=True)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_batches", type=int, default=-1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    row_path = Path(args.result_json).resolve()
    row = json.loads(row_path.read_text())
    model = row.get("model")
    if model not in {"official_base", "mrmsgf"}:
        raise ValueError(f"Unsupported controlled model: {model!r}")
    run_args = SimpleNamespace(
        model=model,
        mr_variant=row.get("mr_variant") or "full",
        graph_topk=8,
        reliability_beta=1.0,
        disable_overlap_reliability=bool(row.get("disable_overlap_reliability", False)),
        disable_readout_reliability=bool(row.get("disable_readout_reliability", False)),
    )
    mask_path = Path(row["mask_path"])
    datamodule, torch_dataset, adjacency, original_mask, dataset, data, input_mask = make_datamodule(mask_path, args.batch_size)
    predictor = build_predictor(run_args, torch_dataset, adjacency)
    checkpoints = sorted((row_path.parent / row["tag"]).glob("*.ckpt"))
    if len(checkpoints) != 1:
        raise ValueError(f"Expected one checkpoint for {row['tag']}, found {len(checkpoints)}")
    predictor.load_state_dict(torch.load(checkpoints[0], map_location="cpu")["state_dict"])
    device = torch.device(args.device)
    predictor.to(device).eval()

    torch_dataset.set_mask(original_mask)
    torch_dataset.set_data(dataset.numpy())
    torch_dataset.add_covariate("x", data, "t n f", add_to_input_map=True, preprocess=True)
    torch_dataset.add_scaler("x", torch_dataset.scalers["target"])
    torch_dataset.add_covariate("input_mask", input_mask, "t n f", add_to_input_map=True)

    abs_sum = sq_sum = count = None
    n_batches = 0
    started = time.perf_counter()
    with torch.no_grad():
        for batch in datamodule.test_dataloader():
            if args.max_batches >= 0 and n_batches >= args.max_batches:
                break
            batch = batch.to(device)
            target, prediction, target_mask = predictor.predict_batch(batch, return_target=True)
            mask = target_mask.float()
            absolute = ((prediction - target).abs() * mask).sum(dim=(0, 2, 3))
            squared = ((prediction - target).square() * mask).sum(dim=(0, 2, 3))
            valid = mask.sum(dim=(0, 2, 3))
            if abs_sum is None:
                abs_sum, sq_sum, count = absolute, squared, valid
            else:
                abs_sum, sq_sum, count = abs_sum + absolute, sq_sum + squared, count + valid
            n_batches += 1
    if abs_sum is None:
        raise RuntimeError("No test batch was evaluated.")
    elapsed = time.perf_counter() - started
    output = {
        "result_json": str(row_path), "checkpoint": str(checkpoints[0]),
        "model": model, "mask_sha256": row["mask_sha256"], "data_sha256": row["data_sha256"],
        "n_test_batches": n_batches,
        "n_model_parameters": int(sum(parameter.numel() for parameter in predictor.model.parameters())),
        "inference_seconds": elapsed,
        "seconds_per_batch": elapsed / n_batches,
        "per_horizon": [
            {"horizon": idx + 1, "n_values": int(count[idx].item()),
             "mae": float((abs_sum[idx] / count[idx].clamp_min(1.0)).cpu()),
             "rmse": float(torch.sqrt(sq_sum[idx] / count[idx].clamp_min(1.0)).cpu())}
            for idx in range(abs_sum.numel())
        ],
    }
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
