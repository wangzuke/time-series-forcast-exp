#!/usr/bin/env python3
"""Train/evaluate an official baseline under the controlled P/B protocol.

Only the data interface, mask generator and experiment loop live here.  The
forecasting networks themselves are imported directly from the authors'
cloned repositories in ``exp/baselines``.  This is intentionally one-process
per-run: BiTGraph and S4M contain hard-coded ``.cuda()`` calls, so a process
is pinned to exactly one visible GPU via ``CUDA_VISIBLE_DEVICES``.
"""
from __future__ import annotations

import argparse
import copy
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
from torch.utils.data import DataLoader

from protocol import EXP_ROOT, ShiftForecastDataset, atomic_json_dump, atomic_torch_save, metrics_sums
from s4m_ablation import apply_s4m_ablation


ROOT = EXP_ROOT.parents[2]
BASELINES = EXP_ROOT / "baselines"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def git_revision(path: pathlib.Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def s4m_config(dataset: str, channels: int, e_layers: int = 8) -> SimpleNamespace:
    """S4M architecture with the paper's dataset-specific hidden widths.

    Appendix D.3 specifies width 512 for Electricity/Traffic and 256 for
    ETTh1/Weather.
    Eight blocks is one of the paper's validation-selected candidates and the
    value in the authors' released Weather command.
    """
    hidden = 512 if dataset in {"Electricity", "Traffic"} else 256
    return SimpleNamespace(
        dropout=0.1, mask=False, d_var=channels, classification=0, plot=0,
        d_model=hidden, d_ff=hidden, en_conv_hidden_size=hidden, e_layers=e_layers, W=6,
        input_keep_prob=0.9, output_keep_prob=0.9, short_len=32, n=10,
        en_rnn_hidden_sizes=[20, 32], factor=1, output_attention=False,
        n_heads=8, K=10, momentum=0.99, topM=100, topK=20, M=50,
        per_mem_size=5, memory_size=200, thres1=0.95, thres2=0.6,
        # Preserve the released queue rule, but retain its runtime state on
        # CUDA to remove the released implementation's per-batch CPU detour.
        memory_device="cuda", is_training=1,
    )


def ctf_config(dataset: str, channels: int, pred_len: int) -> SimpleNamespace:
    """Official CTF model arguments for the closest released benchmark scripts.

    ``keep_prob=1`` is deliberate: the released ETTh1 and Weather scripts use
    it.  The paper's 0.4 *masking ratio* belongs to its separate SolarWind
    robustness ablation, not to these two benchmark commands.
    """
    if dataset == "ETTh1":
        d_model, d_ff, patch = 256, 1024, 24
        globals_ = 2
    elif dataset == "Weather":
        # The authors use d_model=128 on Weather; our controlled regular
        # version has no heterogeneous sampling, hence equal 12-step patches.
        d_model, d_ff, patch = 128, 128, 12
        globals_ = 1
    else:
        # This is only attempted after an explicit CTF GPU smoke test.
        d_model, d_ff, patch = 128, 256, 24
        globals_ = 1
    return SimpleNamespace(
        task_name="long_term_forecast", features="M", seq_len=96,
        pred_len=pred_len, use_norm=1, patch_lens=[patch] * channels,
        sampling_periods=[1.0] * channels, num_global_tokens=globals_,
        keep_prob=1.0, d_model=d_model, d_ff=d_ff, n_heads=8, e_layers=2,
        factor=3, dropout=0.1, activation="gelu", batch_size=32,
        # Local extension: preserves original B=1 semantics while allowing
        # different missing masks within one training batch.
        batched_missing=True,
        enc_in=channels, dec_in=channels, c_out=channels,
    )


def build_model(
    name: str, dataset: str, channels: int, seq_len: int, pred_len: int, device: torch.device,
    s4m_e_layers: int = 8,
) -> nn.Module:
    if name == "BiTGraph":
        repository = BASELINES / "BiTGraph"
        sys.path.insert(0, str(repository))
        from models.BiaTCGNet.BiaTCGNet import Model as OfficialModel
        model = OfficialModel(
            True, True, 2, channels, [2, 3, 6, 7], "cuda:0", predefined_A=None,
            # ICLR'24 Sec. 5.1 fixes three BiTGraph blocks and top-k=10.
            # These two values differ from the released main.py defaults.
            dropout=0.3, subgraph_size=10, node_dim=3, dilation_exponential=1,
            conv_channels=8, residual_channels=8, skip_channels=16,
            end_channels=32, seq_length=seq_len, in_dim=1, out_len=pred_len,
            out_dim=1, layers=3, propalpha=0.05, tanhalpha=3,
            layer_norm_affline=True,
        )
    elif name == "ChannelTokenFormer":
        repository = BASELINES / "ChannelTokenFormer"
        sys.path.insert(0, str(repository))
        from models.ChannelTokenFormer_missing import Model as OfficialModel
        model = OfficialModel(ctf_config(dataset, channels, pred_len))
    elif name == "S4M":
        repository = BASELINES / "S4M" / "s4m"
        # S4M imports both top-level ``model`` and bundled top-level ``S4``.
        sys.path.insert(0, str(repository))
        sys.path.insert(0, str(repository / "model"))
        # The authors' optional compiled structured-kernel package is exposed
        # as ``extensions`` from this bundled S4 root.  Without this entry
        # S4M silently falls back to its documented slow Python kernels.
        sys.path.insert(0, str(repository / "model" / "S4" / "s4"))
        from model.S4M import Model as OfficialModel
        model = OfficialModel(s4m_config(dataset, channels, e_layers=s4m_e_layers))
    else:
        raise ValueError(name)
    return model.to(device).float()


def optimizer_for(
    name: str, dataset: str, model: nn.Module, s4m_optimizer: str = "paper_adam",
) -> torch.optim.Optimizer:
    if name == "S4M":
        learning_rate = {"Electricity": 0.01, "Traffic": 0.01, "ETTh1": 0.005, "Weather": 0.001}[dataset]
        if s4m_optimizer == "official_sgd":
            # This is the non-commented optimizer in the authors' released
            # experiments/exp_pretrain1.py.  It is explicit because the
            # paper text says Adam, whereas the executable entrypoint says
            # SGD; no silent substitution is permitted in a comparison.
            return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)
        # Appendix D.3 reports Adam with dataset-specific learning rates.
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    if name == "BiTGraph":
        return torch.optim.Adam(model.parameters(), lr=1e-3)
    # CTF Appendix A.2: 1e-4 for ETT1 and 1e-2 for Weather.  Electricity
    # has no released CTF command; 1e-3 is an explicitly labelled exploratory
    # setting for the supplementary controlled protocol.
    return torch.optim.Adam(model.parameters(), lr={"ETTh1": 1e-4, "Weather": 1e-2, "Electricity": 1e-3, "Traffic": 1e-3}[dataset])


def s4m_metadata(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reproduce the feature values passed by the official Dataset_Custom4."""
    length = x.size(1)
    index = torch.arange(length, dtype=x.dtype, device=x.device).view(1, length, 1)
    max_position = x.argmax(dim=1, keepdim=True)
    min_position = x.argmin(dim=1, keepdim=True)
    max_idx = (max_position - index).abs()
    min_idx = (min_position - index).abs()
    max_value = x.max(dim=1, keepdim=True).values.expand_as(x)
    min_value = x.min(dim=1, keepdim=True).values.expand_as(x)
    return max_idx, min_idx, max_value, min_value


def prediction(name: str, model: nn.Module, batch: dict[str, torch.Tensor], pred_len: int) -> torch.Tensor:
    x, mask = batch["x"], batch["mask"]
    if name == "BiTGraph":
        # Author model expects B,L,N,1 and the corresponding observation mask.
        return model(x.unsqueeze(-1), mask.unsqueeze(-1), 0).squeeze(-1)
    if name == "ChannelTokenFormer":
        # The local batched extension retains independently observed patches
        # per sample and applies an attention padding mask.
        return model(x, None, None, None, mask)
    max_idx, min_idx, max_value, min_value = s4m_metadata(x)
    # The authors' forecasting head is the final pred_len positions of its
    # sequence output; no new forecast head is introduced here.
    return model(x, mask, max_idx, min_idx, max_value, min_value)[:, -pred_len:, :]


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def forward_loss(name: str, model: nn.Module, batch: dict[str, torch.Tensor], pred_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    pred = prediction(name, model, batch, pred_len)
    if pred.shape != batch["y"].shape:
        raise RuntimeError(f"{name} returned {tuple(pred.shape)}, expected {tuple(batch['y'].shape)}")
    if name == "BiTGraph":
        return pred, torch.mean(torch.abs(pred - batch["y"]))
    return pred, torch.mean((pred - batch["y"]).square())


@torch.no_grad()
def evaluate(
    name: str,
    model: nn.Module,
    loader: DataLoader,
    pred_len: int,
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
    max_batches: int,
) -> dict[str, Any]:
    model.eval()
    if name == "S4M":
        model.mem_net.is_training = 0
    loss_sum, loss_count = 0.0, 0
    mae_sum, mse_sum, element_count = 0.0, 0.0, 0
    step_mae_sum = np.zeros(pred_len, dtype=np.float64)
    step_mse_sum = np.zeros(pred_len, dtype=np.float64)
    step_element_count = np.zeros(pred_len, dtype=np.int64)
    for number, batch in enumerate(loader):
        if max_batches > 0 and number >= max_batches:
            break
        batch = to_device(batch, device)
        pred, loss = forward_loss(name, model, batch, pred_len)
        loss_sum += float(loss.item()) * batch["y"].shape[0]
        loss_count += int(batch["y"].shape[0])
        mae, mse, count = metrics_sums(pred, batch["y_raw"], mean, std)
        mae_sum += mae
        mse_sum += mse
        element_count += count
        pred_raw = pred * std.view(1, 1, -1) + mean.view(1, 1, -1)
        raw_error = pred_raw - batch["y_raw"]
        step_mae_sum += raw_error.abs().sum(dim=(0, 2)).detach().cpu().numpy()
        step_mse_sum += raw_error.square().sum(dim=(0, 2)).detach().cpu().numpy()
        step_element_count += raw_error.shape[0] * raw_error.shape[2]
    if not element_count:
        raise RuntimeError("evaluation produced no batches")
    return {
        "objective": loss_sum / loss_count,
        "mae_raw": mae_sum / element_count,
        "mse_raw": mse_sum / element_count,
        "mae_raw_by_step": (step_mae_sum / step_element_count).tolist(),
        "mse_raw_by_step": (step_mse_sum / step_element_count).tolist(),
        "n_windows": loss_count,
        "n_target_elements": element_count,
    }


def checkpoint_payload(name: str, model: nn.Module) -> dict[str, Any]:
    payload: dict[str, Any] = {"model": copy.deepcopy(model.state_dict())}
    if name == "S4M":
        memory = model.mem_net.memory
        payload["memory"] = memory.memory
        payload["repre"] = memory.repre
        payload["num_dic"] = memory.num_dic
    return payload


def restore_checkpoint(name: str, model: nn.Module, path: pathlib.Path, device: torch.device) -> None:
    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["model"], strict=True)
    if name == "S4M":
        memory = model.mem_net.memory
        memory.memory = payload["memory"]
        memory.repre = payload["repre"]
        memory.num_dic = payload["num_dic"]


def parser() -> argparse.ArgumentParser:
    parsed = argparse.ArgumentParser(description=__doc__)
    parsed.add_argument("--baseline", choices=["S4M", "BiTGraph", "ChannelTokenFormer"], required=True)
    parsed.add_argument("--dataset", choices=["ETTh1", "Weather", "Electricity", "Traffic"], required=True)
    parsed.add_argument("--train-family", choices=["P", "B"], required=True)
    parsed.add_argument("--test-family", choices=["P", "B"], required=True)
    parsed.add_argument("--block-len", type=int, required=True)
    parsed.add_argument("--train-seed", type=int, required=True)
    parsed.add_argument("--test-mask-seed", type=int, required=True)
    parsed.add_argument("--seq-len", type=int, default=96)
    parsed.add_argument("--pred-len", type=int, choices=[24, 96], required=True)
    parsed.add_argument("--missing-rate", type=float, default=0.30)
    parsed.add_argument("--mask-balance", choices=["global", "per_channel"], default="global")
    parsed.add_argument("--s4m-e-layers", type=int, choices=[2, 4, 8], default=8,
                        help="S4M block count; Appendix D.3 selects from {2,4,8} on validation data.")
    parsed.add_argument("--s4m-optimizer", choices=["paper_adam", "official_sgd"], default="paper_adam",
                        help="Resolve the S4M paper/released-code optimizer discrepancy explicitly.")
    parsed.add_argument("--s4m-ablation", choices=["none", "no_mask_stream", "no_prototype_retrieval"], default="none")
    parsed.add_argument("--epochs", type=int, default=20)
    parsed.add_argument("--patience", type=int, default=4)
    parsed.add_argument("--batch-size", type=int, default=32)
    parsed.add_argument("--gradient-accumulation-steps", type=int, default=1,
                        help="Accumulate microbatches to preserve the requested effective batch size.")
    parsed.add_argument("--max-train-batches", type=int, default=-1)
    parsed.add_argument("--max-eval-batches", type=int, default=-1)
    parsed.add_argument("--checkpoint-path", type=pathlib.Path, required=True)
    parsed.add_argument("--result-path", type=pathlib.Path, required=True)
    parsed.add_argument("--progress-path", type=pathlib.Path, required=True)
    parsed.add_argument("--eval-only", action="store_true")
    parsed.add_argument("--skip-test", action="store_true",
                        help="Finish after training/validation; intended for isolated hyperparameter selection.")
    return parsed


def gpu_memory_stats(device: torch.device) -> dict[str, float]:
    """Synchronize and capture per-process peak CUDA memory in MiB."""
    torch.cuda.synchronize(device)
    mib = 1024 * 1024
    return {
        "peak_gpu_allocated_mib": float(torch.cuda.max_memory_allocated(device) / mib),
        "peak_gpu_reserved_mib": float(torch.cuda.max_memory_reserved(device) / mib),
    }


def main() -> None:
    args = parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This protocol must run on a CUDA GPU; CUDA_VISIBLE_DEVICES should expose GPU 6 or 7")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    # Reset before model construction so the recorded peak covers model,
    # optimizer, batches, and evaluation for this one isolated process.
    torch.cuda.reset_peak_memory_stats(device)
    set_seed(args.train_seed)
    requested_batch_size = args.batch_size
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient-accumulation-steps must be positive")
    batch_size = requested_batch_size
    train_ds = ShiftForecastDataset(args.dataset, "train", args.seq_len, args.pred_len, args.train_family,
                                    args.missing_rate, args.train_seed, args.block_len, dynamic_epoch=True,
                                    mask_balance=args.mask_balance)
    val_ds = ShiftForecastDataset(args.dataset, "val", args.seq_len, args.pred_len, args.train_family,
                                  args.missing_rate, args.train_seed + 10_000, args.block_len, dynamic_epoch=False,
                                  mask_balance=args.mask_balance)
    test_ds = ShiftForecastDataset(args.dataset, "test", args.seq_len, args.pred_len, args.test_family,
                                   args.missing_rate, args.test_mask_seed, args.block_len, dynamic_epoch=False,
                                   mask_balance=args.mask_balance)
    loader_kwargs = {"num_workers": 0, "pin_memory": True}
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        drop_last=(args.baseline == "S4M" or args.gradient_accumulation_steps > 1), **loader_kwargs
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
    model = build_model(args.baseline, args.dataset, train_ds.n_features, args.seq_len, args.pred_len, device,
                        s4m_e_layers=args.s4m_e_layers)
    if args.baseline == "S4M":
        apply_s4m_ablation(model, args.s4m_ablation)
    mean = torch.as_tensor(train_ds.mean, device=device)
    std = torch.as_tensor(train_ds.std, device=device)
    source_root = BASELINES / ("S4M" if args.baseline == "S4M" else args.baseline)
    meta: dict[str, Any] = {
        "baseline": args.baseline, "dataset": args.dataset, "train_family": args.train_family,
        "test_family": args.test_family, "block_len": args.block_len,
        "train_seed": args.train_seed, "test_mask_seed": args.test_mask_seed,
        "seq_len": args.seq_len, "pred_len": args.pred_len, "missing_rate": args.missing_rate,
        "mask_balance": args.mask_balance,
        "batch_size": batch_size, "microbatch_size": batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": batch_size * args.gradient_accumulation_steps,
        "requested_batch_size": batch_size * args.gradient_accumulation_steps,
        "s4m_ablation": args.s4m_ablation if args.baseline == "S4M" else None,
        "batch_size_constraint": None,
        "device": torch.cuda.get_device_name(device),
        "official_repository": str(source_root), "official_revision": git_revision(source_root),
        "ctf_paper_hyperparameters": (
            {"optimizer": "Adam", "learning_rate": {"ETTh1": 1e-4, "Weather": 1e-2, "Electricity": 1e-3, "Traffic": 1e-3}[args.dataset],
             "max_epochs": 10, "patience": 3, "e_layers": 2, "keep_prob": 1.0,
             "script_basis": ("released ETTh1/Weather commands" if args.dataset != "Electricity"
                              else "exploratory Electricity extension; no released CTF command"),
             "missing_tokenization": "local batched-padding extension; B=1 equivalence checked"}
            if args.baseline == "ChannelTokenFormer" else None
        ),
        "bitgraph_paper_hyperparameters": (
            {"optimizer": "Adam", "learning_rate": 1e-3, "batch_size": 32,
             "blocks": 3, "top_k_neighbors": 10, "max_epochs": 200,
             "script_basis": "ICLR2024 Sec.5.1 + released README command"}
            if args.baseline == "BiTGraph" else None
        ),
        "s4m_paper_hyperparameters": (
            {"optimizer": ("SGD(momentum=0.9,weight_decay=1e-5)" if args.s4m_optimizer == "official_sgd" else "Adam"),
             "optimizer_source": ("released experiments/exp_pretrain1.py" if args.s4m_optimizer == "official_sgd"
                                  else "paper Appendix D.3"),
             "learning_rate": {"Electricity": 0.01, "Traffic": 0.01, "ETTh1": 0.005, "Weather": 0.001}[args.dataset],
             "hidden_width": 512 if args.dataset in {"Electricity", "Traffic"} else 256,
             "batch_size": 16, "e_layers": args.s4m_e_layers,
             "memory_update": "cuda_serial_equivalent"}
            if args.baseline == "S4M" else None
        ),
        "mask_audit": {"train": train_ds.audit(), "val": val_ds.audit(), "test": test_ds.audit()},
        "target_missing_rate": 0.0,
    }
    atomic_json_dump(meta, args.progress_path.with_name(args.progress_path.stem + "_protocol.json"))
    started = time.time()
    try:
        if args.eval_only:
            restore_checkpoint(args.baseline, model, args.checkpoint_path, device)
            training = {"mode": "eval_only", "best_epoch": None, "best_val_objective": None}
        else:
            optimizer = optimizer_for(args.baseline, args.dataset, model, args.s4m_optimizer)
            best_objective, best_epoch, stale = float("inf"), -1, 0
            # S4M's memory bank requires a warm start.  We retain that bank
            # across epochs (equivalent to its --no_renew behaviour), avoiding
            # the upstream entrypoint's reset-without-rewarm bug.
            s4m_warmed = False
            for epoch in range(1, args.epochs + 1):
                train_ds.set_epoch(epoch)
                model.train()
                if args.baseline == "S4M":
                    model.mem_net.is_training = 1
                loss_sum, windows = 0.0, 0
                epoch_batches = len(train_loader)
                if args.max_train_batches > 0:
                    epoch_batches = min(epoch_batches, args.max_train_batches)
                for number, batch in enumerate(train_loader):
                    if number >= epoch_batches:
                        break
                    batch = to_device(batch, device)
                    if args.baseline == "S4M" and not s4m_warmed:
                        max_idx, min_idx, max_value, min_value = s4m_metadata(batch["x"])
                        model.warmup(batch["x"], batch["mask"], max_idx, min_idx, max_value, min_value)
                        s4m_warmed = True
                        continue
                    if number % args.gradient_accumulation_steps == 0:
                        optimizer.zero_grad(set_to_none=True)
                    _, loss = forward_loss(args.baseline, model, batch, args.pred_len)
                    group_start = (number // args.gradient_accumulation_steps) * args.gradient_accumulation_steps
                    group_size = min(args.gradient_accumulation_steps, epoch_batches - group_start)
                    (loss / group_size).backward()
                    # Gradient clipping is not part of the published S4M
                    # configuration, so do not introduce it for that baseline.
                    if args.baseline != "S4M":
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    if ((number + 1) % args.gradient_accumulation_steps == 0
                            or number + 1 == epoch_batches):
                        optimizer.step()
                    loss_sum += float(loss.item()) * batch["y"].shape[0]
                    windows += int(batch["y"].shape[0])
                val_metrics = evaluate(args.baseline, model, val_loader, args.pred_len, device, mean, std, args.max_eval_batches)
                if val_metrics["objective"] < best_objective:
                    best_objective, best_epoch, stale = float(val_metrics["objective"]), epoch, 0
                    atomic_torch_save(checkpoint_payload(args.baseline, model), args.checkpoint_path)
                else:
                    stale += 1
                progress = {
                    **meta, "status": "running", "epoch": epoch, "elapsed_sec": time.time() - started,
                    "train_objective": loss_sum / max(1, windows), "train_windows": windows,
                    "val": val_metrics, "best_epoch": best_epoch, "best_val_objective": best_objective,
                    "stale_epochs": stale,
                    **gpu_memory_stats(device),
                }
                atomic_json_dump(progress, args.progress_path)
                print(json.dumps(progress, ensure_ascii=False), flush=True)
                if stale >= args.patience:
                    break
            restore_checkpoint(args.baseline, model, args.checkpoint_path, device)
            training = {"mode": "train", "best_epoch": best_epoch, "best_val_objective": best_objective}
        result = {**meta, **training, "status": "complete", "elapsed_sec": time.time() - started,
                  **gpu_memory_stats(device)}
        if args.skip_test:
            result["selection_only"] = True
        else:
            result["test"] = evaluate(args.baseline, model, test_loader, args.pred_len, device, mean, std,
                                      args.max_eval_batches)
        atomic_json_dump(result, args.result_path)
        atomic_json_dump(result, args.progress_path)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    except Exception as error:
        failed = {**meta, "status": "failed", "elapsed_sec": time.time() - started, **gpu_memory_stats(device),
                  "error": repr(error), "traceback": traceback.format_exc()}
        atomic_json_dump(failed, args.progress_path)
        atomic_json_dump(failed, args.result_path)
        raise


if __name__ == "__main__":
    main()
