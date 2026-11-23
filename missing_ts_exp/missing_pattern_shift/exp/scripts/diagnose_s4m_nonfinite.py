#!/usr/bin/env python3
"""Locate the first non-finite value in a short S4M training replay."""
from __future__ import annotations

import argparse
import json
import pathlib

import torch
from torch.utils.data import DataLoader

from protocol import EXP_ROOT, ShiftForecastDataset, atomic_json_dump
from run_official_baseline import build_model, forward_loss, optimizer_for, set_seed, s4m_metadata, to_device


def finite(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor).all().item())


def named_nonfinite(module: torch.nn.Module) -> str | None:
    for name, parameter in module.named_parameters():
        if not finite(parameter):
            return name
    return None


def tensor_summary(tensor: torch.Tensor | None) -> dict[str, float | bool] | None:
    """Compact finite/range audit without retaining an activation graph."""
    if tensor is None:
        return None
    detached = tensor.detach()
    return {
        "finite": finite(detached),
        "abs_max": float(detached.abs().max().item()),
        "mean": float(detached.mean().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Electricity", choices=["ETTh1", "Weather", "Electricity"])
    parser.add_argument("--e-layers", type=int, default=2, choices=[2, 4, 8])
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--optimizer", choices=["adam", "official_sgd"], default="adam",
                        help="`official_sgd` reproduces the active optimizer line in experiments/exp_pretrain1.py.")
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--check-interval", type=int, default=64,
                        help="Full parameter/gradient scan interval; scalar loss is checked every step.")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    set_seed(2024)
    dataset = ShiftForecastDataset(args.dataset, "train", 96, 96, "P", 0.5, 2024, 24,
                                   dynamic_epoch=True, mask_balance="per_channel")
    dataset.set_epoch(1)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True, num_workers=0, pin_memory=True)
    model = build_model("S4M", args.dataset, dataset.n_features, 96, 96, device, s4m_e_layers=args.e_layers)
    if args.optimizer == "official_sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-5)
    else:
        optimizer = optimizer_for("S4M", args.dataset, model, "paper_adam")
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
    model.train()
    model.mem_net.is_training = 1
    records: list[dict[str, object]] = []
    failure: dict[str, object] | None = None
    warmed = False
    for step, batch in enumerate(loader):
        if step >= args.max_steps:
            break
        batch = to_device(batch, device)
        if not warmed:
            max_idx, min_idx, max_value, min_value = s4m_metadata(batch["x"])
            model.warmup(batch["x"], batch["mask"], max_idx, min_idx, max_value, min_value)
            warmed = True
            continue
        optimizer.zero_grad(set_to_none=True)
        prediction, loss = forward_loss("S4M", model, batch, 96)
        full_scan = step % args.check_interval == 0
        record: dict[str, object] = {
            "step": step,
            "x_finite": finite(batch["x"]), "mask_finite": finite(batch["mask"]),
            "prediction_finite": finite(prediction), "loss_finite": finite(loss),
            "loss": float(loss.detach().item()),
            "full_scan": full_scan,
        }
        if not record["prediction_finite"] or not record["loss_finite"]:
            memory = model.mem_net.memory
            failure = {
                **record, "stage": "forward", "parameter_nonfinite": named_nonfinite(model),
                "memory_repre": tensor_summary(memory.repre),
                "memory_entries": len(memory.memory) if memory.memory is not None else 0,
                "decay1_weight": tensor_summary(model.decay1.weight),
                "decay2_weight": tensor_summary(model.decay2.weight),
            }
            records.append(record)
            break
        loss.backward()
        if full_scan:
            bad_gradient = next((name for name, parameter in model.named_parameters()
                                 if parameter.grad is not None and not finite(parameter.grad)), None)
            record["gradient_nonfinite"] = bad_gradient
            if bad_gradient is not None:
                failure = {**record, "stage": "backward"}
                records.append(record)
                break
        optimizer.step()
        if full_scan:
            record["parameter_nonfinite_after"] = named_nonfinite(model)
        records.append(record)
        if full_scan and record["parameter_nonfinite_after"] is not None:
            failure = {**record, "stage": "optimizer_step"}
            break
    result = {
        "dataset": args.dataset, "e_layers": args.e_layers, "learning_rate": args.learning_rate,
        "optimizer": args.optimizer,
        "max_steps": args.max_steps, "completed_steps": len(records), "failure": failure,
        "last_records": records[-10:],
    }
    atomic_json_dump(result, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
