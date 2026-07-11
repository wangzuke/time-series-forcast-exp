#!/usr/bin/env python3
"""Analyze MissTSM route/fuse/gate diagnostics from a saved checkpoint.

The script rebuilds the pipeline from the checkpoint config, enables the
lightweight diagnostic hooks in MissTSM, runs a few test batches, and writes
aggregated route/gate/path statistics to JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.training.pipelines import PipelineConfig, build_pipeline  # noqa: E402
from src.training.run_forecast import get_loader  # noqa: E402
from src.data.timefeatures import time_feature_dim  # noqa: E402
from src.utils.constants import DATASETS  # noqa: E402


def _to_float(value):
    if torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.item())
        return [float(x) for x in value.flatten().tolist()]
    if isinstance(value, (int, float)):
        return float(value)
    return value


def _cfg_from_checkpoint(raw_cfg: dict) -> PipelineConfig:
    dataset = raw_cfg["dataset"]
    meta = DATASETS[dataset]
    return PipelineConfig(
        method=raw_cfg.get("method", "misstsm"),
        predictor=raw_cfg.get("predictor", "iTransformer"),
        impute_strategy=raw_cfg.get("impute", "none"),
        dataset=dataset,
        seq_len=int(raw_cfg.get("seq_len", 96)),
        pred_len=int(raw_cfg.get("pred_len", 96)),
        n_channels=int(meta["n_features"]),
        time_feat_dim=time_feature_dim(meta["freq"]),
        misstsm_variant=raw_cfg.get("misstsm_variant", "full"),
        group_entropy_weight=float(raw_cfg.get("group_entropy_weight", 0.0)),
        missing_type=raw_cfg.get("missing_type", "none"),
        missing_rate=float(raw_cfg.get("missing_rate", 0.0)),
        mask_aware=raw_cfg.get("mask_aware", "none"),
        coifnet_hidden=int(raw_cfg.get("coifnet_hidden", 256)),
        coifnet_input_form=raw_cfg.get("coifnet_input_form", "x_cat_mask"),
        coifnet_embed_type=raw_cfg.get("coifnet_embed_type", "shared"),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--max_batches", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    raw_cfg = ckpt["config"]
    cfg = _cfg_from_checkpoint(raw_cfg)
    model = build_pipeline(cfg).to(args.device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    if not hasattr(model, "model") or not hasattr(model.model, "enable_diagnostics"):
        raise RuntimeError("checkpoint does not expose MissTSM diagnostics")
    model.model.enable_diagnostics(True)

    loader, _ = get_loader(
        cfg.dataset, args.split, cfg.seq_len, cfg.pred_len,
        cfg.missing_type, cfg.missing_rate, int(raw_cfg.get("seed", 2024)),
        args.batch_size, int(raw_cfg.get("num_workers", 2)), shuffle=False,
    )

    sums = defaultdict(float)
    counts = defaultdict(int)
    vector_sums: dict[str, torch.Tensor] = {}
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(args.device, non_blocking=True) for k, v in batch.items()}
            _ = model(batch)
            diag = model.model.diagnostics()
            if not diag:
                continue
            n_batches += 1
            for key, value in diag.items():
                if torch.is_tensor(value) and value.numel() > 1:
                    value = value.detach().cpu().float()
                    if key not in vector_sums:
                        vector_sums[key] = torch.zeros_like(value)
                    vector_sums[key] += value
                else:
                    sums[key] += float(_to_float(value))
                    counts[key] += 1
            if n_batches >= args.max_batches:
                break

    result = {
        "checkpoint": args.checkpoint,
        "config": raw_cfg,
        "split": args.split,
        "batches_analyzed": n_batches,
        "scalars": {k: sums[k] / max(1, counts[k]) for k in sorted(sums)},
        "vectors": {k: _to_float(v / max(1, n_batches)) for k, v in sorted(vector_sums.items())},
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
