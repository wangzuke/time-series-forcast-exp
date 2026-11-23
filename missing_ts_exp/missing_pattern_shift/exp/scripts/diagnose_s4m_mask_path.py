#!/usr/bin/env python3
"""Trace whether S4M consumes missing-value masks in the controlled protocol.

This is a non-training diagnostic.  It replays a fixed checkpoint on matched
point- and block-missing batches, records P/B differences at every major S4M
boundary, and separates the effects of the zero-filled values from those of
the observation mask.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from collections.abc import Callable
from typing import Any

import torch
from torch.utils.data import DataLoader

from protocol import EXP_ROOT, ShiftForecastDataset, atomic_json_dump
from run_official_baseline import build_model, prediction, restore_checkpoint, s4m_metadata, to_device


def tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "device": str(tensor.device),
        "dtype": str(tensor.dtype),
        "abs_sum": float(tensor.abs().sum().item()),
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std().item()),
        "finite": bool(torch.isfinite(tensor).all().item()),
    }


def delta(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    difference = left - right
    return {
        "l1": float(difference.abs().sum().item()),
        "mean_abs": float(difference.abs().mean().item()),
        "max_abs": float(difference.abs().max().item()),
        "relative_l1_to_left": float(difference.abs().sum().item() / left.abs().sum().clamp_min(1e-12).item()),
    }


def run_and_capture(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    pred_len: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    captured: dict[str, torch.Tensor] = {}
    hooks: list[Any] = []

    def capture(name: str) -> Callable[..., None]:
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            value = output[0] if isinstance(output, tuple) else output
            captured[name] = value.detach().clone()
        return hook

    # These locations cover the hand-off from imputation into memory retrieval,
    # mask embedding, masked S4, and the forecast projection.
    traced_modules = {
        "memory_features": model.mem_net,
        "mask_embedding": model.mask_trans,
        "post_memory_linear": model.linear1,
        "final_layer_norm": model.norm,
        "output_projection": model.out_proj,
    }
    # The first block is mask-aware; later blocks are regular S4 blocks.  Each
    # is captured separately to identify a possible representation collapse.
    traced_modules.update({f"s4_layer_{index}": layer for index, layer in enumerate(model.s4_layers)})
    for name, module in traced_modules.items():
        hooks.append(module.register_forward_hook(capture(name)))
    try:
        with torch.no_grad():
            output = prediction("S4M", model, batch, pred_len).detach().clone()
    finally:
        for hook in hooks:
            hook.remove()
    captured["prediction"] = output
    return output, captured


def make_variant(reference: dict[str, torch.Tensor], *, x: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
    # Metadata is recomputed exactly as the training/evaluation runner does;
    # keeping it coupled to x avoids falsely testing an inconsistent interface.
    variant = {key: value for key, value in reference.items()}
    variant["x"] = x
    variant["mask"] = mask
    return variant


def compare_layers(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {
        name: {"left": tensor_stats(left[name]), "right": tensor_stats(right[name]), "difference": delta(left[name], right[name])}
        for name in left
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--dataset", default="ETTh1", choices=["ETTh1", "Weather", "Electricity"])
    parser.add_argument("--split", default="test", choices=["val", "test"],
                        help="Split used only for the non-training sensitivity screen.")
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--missing-rate", type=float, default=0.5)
    parser.add_argument("--block-len", type=int, default=24)
    parser.add_argument("--mask-seed", type=int, default=9101)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--s4m-e-layers", type=int, choices=[2, 4, 8], default=8)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("S4M diagnostic requires CUDA")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    point = ShiftForecastDataset(args.dataset, args.split, args.seq_len, args.pred_len, "P", args.missing_rate,
                                 args.mask_seed, args.block_len, dynamic_epoch=False, mask_balance="per_channel")
    block = ShiftForecastDataset(args.dataset, args.split, args.seq_len, args.pred_len, "B", args.missing_rate,
                                 args.mask_seed, args.block_len, dynamic_epoch=False, mask_balance="per_channel")
    point_batch = to_device(next(iter(DataLoader(point, batch_size=args.batch_size, shuffle=False))), device)
    block_batch = to_device(next(iter(DataLoader(block, batch_size=args.batch_size, shuffle=False))), device)
    model = build_model("S4M", args.dataset, point.n_features, args.seq_len, args.pred_len, device,
                        s4m_e_layers=args.s4m_e_layers)
    restore_checkpoint("S4M", model, args.checkpoint, device)
    model.eval()
    model.mem_net.is_training = 0

    # Four controlled variants: P, B, P values+B mask, B values+P mask.
    variants = {
        "point": point_batch,
        "block": block_batch,
        "point_values_block_mask": make_variant(point_batch, x=point_batch["x"], mask=block_batch["mask"]),
        "block_values_point_mask": make_variant(block_batch, x=block_batch["x"], mask=point_batch["mask"]),
        # A deliberately large, mask-preserving control: it swaps histories
        # across samples.  If this does not alter a forecast, the issue is
        # broader than sensitivity to the P/B mask construction.
        "point_values_batch_permuted": make_variant(
            point_batch, x=point_batch["x"].roll(shifts=1, dims=0), mask=point_batch["mask"]
        ),
    }
    captures: dict[str, dict[str, torch.Tensor]] = {}
    for name, batch in variants.items():
        _, captures[name] = run_and_capture(model, batch, args.pred_len)
    point_x = point_batch["x"]
    block_x = block_batch["x"]
    point_mask = point_batch["mask"]
    block_mask = block_batch["mask"]
    result = {
        "purpose": "S4M P/B mask-path diagnostic; no parameters were updated",
        "checkpoint": str(args.checkpoint),
        "protocol": {"dataset": args.dataset, "split": args.split, "seq_len": args.seq_len, "pred_len": args.pred_len,
                     "missing_rate": args.missing_rate, "block_len": args.block_len, "mask_seed": args.mask_seed,
                     "batch_size": args.batch_size, "mask_balance": "per_channel", "s4m_e_layers": args.s4m_e_layers},
        "dataset_audit": {"point": point.audit(), "block": block.audit()},
        "input_difference": {
            "x": delta(point_x, block_x),
            "mask": delta(point_mask, block_mask),
            "mask_positions_different": int((point_mask != block_mask).sum().item()),
            "point_missing": int((point_mask == 0).sum().item()),
            "block_missing": int((block_mask == 0).sum().item()),
            "point_metadata": {name: tensor_stats(value) for name, value in zip(
                ("max_idx", "min_idx", "max_value", "min_value"), s4m_metadata(point_x))},
            "block_metadata": {name: tensor_stats(value) for name, value in zip(
                ("max_idx", "min_idx", "max_value", "min_value"), s4m_metadata(block_x))},
        },
        "layer_differences": {
            "point_vs_block": compare_layers(captures["point"], captures["block"]),
            "point_vs_point_values_block_mask": compare_layers(captures["point"], captures["point_values_block_mask"]),
            "point_vs_block_values_point_mask": compare_layers(captures["point"], captures["block_values_point_mask"]),
            "point_vs_point_values_batch_permuted_control": compare_layers(
                captures["point"], captures["point_values_batch_permuted"]
            ),
        },
        "interpretation_thresholds": {
            "relative_l1": "A value near zero means the layer output is insensitive to the stated perturbation.",
            "variants": "P vs B changes both zero-filled values and mask; the other two comparisons isolate mask-only and value-only changes."
        },
    }
    atomic_json_dump(result, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
