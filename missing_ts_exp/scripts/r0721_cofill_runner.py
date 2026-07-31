#!/usr/bin/env python3
"""0721 CoFILL runner using canonical HDF5 data and shared observed masks."""

from __future__ import annotations

import argparse
import json
import pathlib
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset


COFILL = pathlib.Path("/data/wangzuke/time-series-forecast-exp/external_repro/CoFILL")
if str(COFILL) not in sys.path:
    sys.path.insert(0, str(COFILL))

from main_model import CoFILL_MetrLA, CoFILL_PemsBAY  # noqa: E402
from utils import evaluate, get_block_mask, get_randmask, train  # noqa: E402


DATASET_META = {
    "Metr": {"nodes": 207, "adj_file": "metr-la", "model": CoFILL_MetrLA},
    "PEMS": {"nodes": 325, "adj_file": "pems-bay", "model": CoFILL_PemsBAY},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="0721 CoFILL canonical-mask runner")
    parser.add_argument("--dataset", choices=["Metr", "PEMS"], required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--mask_path", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--effective_batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--task", choices=["imputation"], default="imputation")
    parser.add_argument("--config", default=str(COFILL / "config" / "traffic.yaml"))
    parser.add_argument("--nsample", type=int, default=5)
    parser.add_argument("--diffusion_steps", type=int, default=None)
    parser.add_argument("--target_strategy", choices=["external", "block", "random"], default="external")
    parser.add_argument("--missing_type", choices=["block_t", "block_st", "point"], default=None)
    parser.add_argument("--target_missing_rate", type=float, default=None)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


class CanonicalMaskDataset(Dataset):
    def __init__(
        self,
        data_path: pathlib.Path,
        mask_path: pathlib.Path,
        dataset: str,
        mode: str,
        eval_length: int = 24,
        val_len: float = 0.1,
        test_len: float = 0.2,
        is_interpolate: bool = True,
        target_strategy: str = "external",
    ):
        self.dataset = dataset
        self.mode = mode
        self.eval_length = eval_length
        self.is_interpolate = is_interpolate
        self.target_strategy = target_strategy

        df = pd.read_hdf(data_path)
        values = df.fillna(0).values.astype(np.float32)
        base_observed_mask = np.isfinite(df.values).astype(np.float32)
        external_observed_mask = np.load(mask_path)
        if external_observed_mask.ndim == 3:
            external_observed_mask = external_observed_mask[:, :, 0]
        external_observed_mask = external_observed_mask.astype(np.float32)
        if external_observed_mask.shape != values.shape:
            raise ValueError(
                f"mask shape {external_observed_mask.shape} != data shape {values.shape}"
            )

        data_len = len(df)
        train_data = values[: int(data_len * 0.7)]
        self.train_mean = np.mean(train_data, axis=0).astype(np.float32)
        self.train_std = np.std(train_data, axis=0).astype(np.float32)
        self.train_std[self.train_std == 0] = 1.0

        normalized = ((values - self.train_mean) / self.train_std).astype(np.float32)
        observed_mask = base_observed_mask.astype(np.float32)
        gt_mask = (external_observed_mask * observed_mask).astype(np.float32)

        val_start = int((1 - val_len - test_len) * data_len)
        test_start = int((1 - test_len) * data_len)
        if mode == "train":
            sl = slice(0, val_start)
        elif mode == "valid":
            sl = slice(val_start, test_start)
        elif mode == "test":
            sl = slice(test_start, data_len)
        else:
            raise ValueError(mode)

        self.observed_data = (normalized[sl] * observed_mask[sl]).astype(np.float32)
        self.observed_mask = observed_mask[sl].astype(np.float32)
        self.gt_mask = gt_mask[sl].astype(np.float32)

        current_length = len(self.observed_mask) - eval_length + 1
        self.use_index: list[int]
        self.cut_length: list[int]
        if mode == "test":
            n_sample = len(self.observed_data) // eval_length
            c_index = np.arange(0, eval_length * n_sample, eval_length)
            self.use_index = c_index.tolist()
            self.cut_length = [0] * len(c_index)
            if len(self.observed_data) % eval_length != 0:
                self.use_index.append(current_length - 1)
                self.cut_length.append(eval_length - len(self.observed_data) % eval_length)
        else:
            self.use_index = np.arange(current_length).tolist()
            self.cut_length = [0] * len(self.use_index)

    def __len__(self) -> int:
        return len(self.use_index)

    def __getitem__(self, org_index: int) -> dict:
        index = self.use_index[org_index]
        ob_data = self.observed_data[index : index + self.eval_length]
        ob_mask = self.observed_mask[index : index + self.eval_length]
        gt_mask = self.gt_mask[index : index + self.eval_length]

        ob_mask_t = torch.tensor(ob_mask).float()
        gt_mask_t = torch.tensor(gt_mask).float()
        if self.target_strategy == "external":
            cond_mask = gt_mask_t
        elif self.target_strategy == "block":
            cond_mask = get_block_mask(ob_mask_t, target_strategy="block")
        elif self.target_strategy == "random":
            cond_mask = get_randmask(ob_mask_t)
        else:
            raise ValueError(self.target_strategy)

        s = {
            "observed_data": ob_data,
            "observed_mask": ob_mask,
            "gt_mask": gt_mask,
            "timepoints": np.arange(self.eval_length),
            "cut_length": self.cut_length[org_index],
            "cond_mask": cond_mask.numpy(),
        }

        if self.is_interpolate:
            tmp_data = torch.tensor(ob_data).float() * cond_mask
            for t in range(1, tmp_data.shape[0]):
                tmp_data[t] = torch.where(cond_mask[t] == 0, tmp_data[t - 1], tmp_data[t])
            s["coeffs"] = tmp_data.numpy()
        else:
            s["coeffs"] = None
        return s


def build_loaders(args: argparse.Namespace, config: dict):
    common = dict(
        data_path=pathlib.Path(args.data_path),
        mask_path=pathlib.Path(args.mask_path),
        dataset=args.dataset,
        eval_length=24,
        is_interpolate=bool(config["model"]["use_guide"]),
        target_strategy=args.target_strategy,
    )
    train_dataset = CanonicalMaskDataset(mode="train", **common)
    valid_dataset = CanonicalMaskDataset(mode="valid", **common)
    test_dataset = CanonicalMaskDataset(mode="test", **common)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
    )
    scaler = torch.from_numpy(train_dataset.train_std).to(args.device).float()
    mean_scaler = torch.from_numpy(train_dataset.train_mean).to(args.device).float()
    return train_loader, valid_loader, test_loader, scaler, mean_scaler


def main() -> int:
    args = parse_args()
    run_dir = pathlib.Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = yaml.safe_load(pathlib.Path(args.config).read_text())
    config["seed"] = args.seed
    config["train"]["batch_size"] = args.batch_size
    config["train"]["epochs"] = args.epochs
    config["diffusion"]["adj_file"] = DATASET_META[args.dataset]["adj_file"]
    if args.diffusion_steps is not None:
        config["diffusion"]["num_steps"] = args.diffusion_steps
    config["model"]["target_strategy"] = args.target_strategy

    mask = np.load(args.mask_path)
    actual_missing = 1.0 - float(mask.mean())
    metadata = {
        "model": "CoFILL",
        "variant": "original_imputation_0721_external_mask",
        "dataset": args.dataset,
        "data_path": args.data_path,
        "mask_path": args.mask_path,
        "actual_missing_rate": actual_missing,
        "target_missing_rate": args.target_missing_rate,
        "missing_type": args.missing_type,
        "mask_convention": "1=observed,0=missing",
        "task": args.task,
        "batch_size": args.batch_size,
        "effective_batch_size": args.effective_batch_size or args.batch_size,
        "epochs": args.epochs,
        "nsample": args.nsample,
        "diffusion_steps": config["diffusion"]["num_steps"],
        "target_strategy": args.target_strategy,
        "seed": args.seed,
        "config": config,
    }
    (run_dir / "cofill_0721_config.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False)
    )
    print("[cofill_h5]", json.dumps({k: v for k, v in metadata.items() if k != "config"}, ensure_ascii=False))

    train_loader, valid_loader, test_loader, scaler, mean_scaler = build_loaders(args, config)
    model_class = DATASET_META[args.dataset]["model"]
    model = model_class(config, args.device).to(args.device)

    if args.dry_run:
        print("[cofill_metrics]", json.dumps({"status": "dry_run", "MAE": "", "MSE": "", "CRPS": ""}))
        return 0

    start_time = time.time()
    train(model, config["train"], train_loader, valid_loader=valid_loader, foldername=str(run_dir))
    train_time_sec = time.time() - start_time

    eval_start = time.time()
    evaluate(
        model,
        test_loader,
        nsample=args.nsample,
        scaler=scaler,
        mean_scaler=mean_scaler,
        foldername=str(run_dir),
    )
    inference_time_sec = time.time() - eval_start

    result_path = run_dir / f"result_nsample{args.nsample}.pk"
    with result_path.open("rb") as f:
        rmse, mae, crps = pickle.load(f)
    metrics = {
        "status": "finished",
        "MAE": float(mae),
        "MSE": float(rmse) ** 2,
        "RMSE": float(rmse),
        "CRPS": float(crps),
        "train_time_sec": round(train_time_sec, 3),
        "inference_time_sec": round(inference_time_sec, 3),
        "checkpoint_path": str((run_dir / "model.pth").relative_to(pathlib.Path("/data/wangzuke/time-series-forecast-exp"))),
        "prediction_path": str((run_dir / f"generated_outputs_nsample{args.nsample}.pk").relative_to(pathlib.Path("/data/wangzuke/time-series-forecast-exp"))),
        "notes": f"diffusion_steps={config['diffusion']['num_steps']}; nsample={args.nsample}; target_strategy={args.target_strategy}",
    }
    print("[cofill_metrics]", json.dumps(metrics, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
