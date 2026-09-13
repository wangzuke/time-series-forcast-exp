#!/usr/bin/env python3
"""Run the unmodified official BiTGraph model on the controlled Block-T protocol.

The author repository's ``main.py`` only constructs random point masks.  This
adapter intentionally leaves the author model untouched and replaces only that
data interface: it reuses the exact METR-LA window, split, observed mask, and
unmasked test target used by the controlled HD-TTS/MR-MSGF comparison.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pathlib
import random
import sys

import numpy as np
import torch

from .run_mrmsgf_official_controlled import ROOT, make_datamodule


BITGRAPH_ROOT = ROOT / "external_repro" / "official" / "bitgraph"
if str(BITGRAPH_ROOT) not in sys.path:
    sys.path.insert(0, str(BITGRAPH_ROOT))

from models.BiaTCGNet.BiaTCGNet import Model as OfficialBiTGraphModel  # noqa: E402


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(n_nodes: int, horizon: int) -> OfficialBiTGraphModel:
    """Instantiate exactly the architecture arguments used in upstream main.py."""
    return OfficialBiTGraphModel(
        True, True, 2, n_nodes, [2, 3, 6, 7], "cuda:0", predefined_A=None,
        dropout=0.3, subgraph_size=5, node_dim=3, dilation_exponential=1,
        conv_channels=8, residual_channels=8, skip_channels=16,
        end_channels=32, seq_length=24, in_dim=1, out_len=horizon,
        out_dim=1, layers=2, propalpha=0.05, tanhalpha=3,
        layer_norm_affline=True,
    ).cuda()


def raw_values(batch: dict[str, torch.Tensor], prediction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scaler = batch["transform"]["y"]
    # TSL stores batch transforms on CPU while the official model runs on GPU.
    # Apply the same affine inverse explicitly on the prediction device.
    scale = scaler.scale.to(prediction.device, prediction.dtype)
    bias = scaler.bias.to(prediction.device, prediction.dtype)
    # ``y`` emitted by the TSL dataset is already in raw units.  The model,
    # like the upstream implementation, predicts a standardized target.
    target = batch["y"].to(prediction.device, prediction.dtype)
    return prediction * (scale + 1e-8) + bias, target


def standardized_target(batch: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    """Match the upstream loader, which standardizes y before its L1 loss."""
    scaler = batch["transform"]["y"]
    scale = scaler.scale.to(device)
    bias = scaler.bias.to(device)
    target = batch["y"].to(device)
    return (target - bias) / (scale + 1e-8)


def masked_mae(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(prediction.dtype)
    return (torch.abs(prediction - target) * weight).sum() / weight.sum().clamp_min(1.0)


@torch.no_grad()
def evaluate(model: OfficialBiTGraphModel, loader, *, target_mask: bool) -> tuple[float, float]:
    model.eval()
    abs_error = 0.0
    square_error = 0.0
    count = 0.0
    for batch_index, batch in enumerate(loader):
        x = batch["x"].cuda(non_blocking=True)
        input_mask = batch["input_mask"].cuda(non_blocking=True)
        prediction = model(x, input_mask, batch_index)
        prediction, target = raw_values(batch, prediction)
        mask = batch["mask"] if target_mask else torch.ones_like(batch["mask"])
        mask = mask.cuda(non_blocking=True).to(prediction.dtype)
        abs_error += float((torch.abs(prediction - target) * mask).sum().item())
        square_error += float(((prediction - target).square() * mask).sum().item())
        count += float(mask.sum().item())
    return abs_error / count, square_error / count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_path", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    torch.set_num_threads(1)  # Matches the author training script.
    set_seed(args.seed)
    mask_path = pathlib.Path(args.mask_path).resolve()
    out_dir = pathlib.Path(args.out_dir)
    run_dir = out_dir / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)

    datamodule, torch_dataset, _, original_mask, dataset, data, mask = make_datamodule(mask_path, args.batch_size)
    model = build_model(torch_dataset.n_nodes, torch_dataset.horizon)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # Upstream main.py.

    best_val_mae = float("inf")
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        for batch in datamodule.train_dataloader():
            x = batch["x"].cuda(non_blocking=True)
            input_mask = batch["input_mask"].cuda(non_blocking=True)
            target = standardized_target(batch, x.device)
            target_mask = batch["mask"].cuda(non_blocking=True)
            prediction = model(x, input_mask, 0)  # Upstream train loop passes k=0.
            loss = masked_mae(prediction, target, target_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_mae, _ = evaluate(model, datamodule.val_dataloader(), target_mask=True)
        print(json.dumps({"epoch": epoch, "val_mae": val_mae}), flush=True)
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("No BiTGraph checkpoint was selected.")
    model.load_state_dict(best_state)
    torch.save(best_state, run_dir / "best.pth")

    # Mirror the HD-TTS controlled runner's unmasked test pass while preserving
    # the artificial Block-T input_mask for the model.
    torch_dataset.set_mask(original_mask)
    torch_dataset.set_data(dataset.numpy())
    torch_dataset.add_covariate("x", data, "t n f", add_to_input_map=True, preprocess=True)
    torch_dataset.add_scaler("x", torch_dataset.scalers["target"])
    torch_dataset.add_covariate("input_mask", mask, "t n f", add_to_input_map=True)
    test_mae, test_mse = evaluate(model, datamodule.test_dataloader(), target_mask=False)

    row = dict(
        tag=args.tag,
        model="bitgraph_official_architecture",
        protocol="unified_hdtts_block_t",
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        mask_path=str(mask_path),
        mask_sha256=sha256(mask_path),
        artificial_missing_rate=float(1.0 - np.load(mask_path).mean()),
        data_sha256=sha256(ROOT / "dataset" / "METR-LA" / "metr-la.h5"),
        official_bitgraph_commit="4f1d05bcc20bb3f5084bd1facc995478f84a40ed",
        model_hparams=dict(gcn_depth=2, kernel_set=[2, 3, 6, 7], dropout=0.3,
                           subgraph_size=5, node_dim=3, conv_channels=8,
                           residual_channels=8, skip_channels=16, end_channels=32,
                           layers=2, window=24, horizon=12),
        optimizer="Adam(lr=1e-3)",
        best_val_mae=best_val_mae,
        unmasked_test=dict(test_mae_unmasked=test_mae, test_mse_unmasked=test_mse),
    )
    path = out_dir / f"{args.tag}.json"
    path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
