"""Controlled MR-MSGF experiment on the unmodified HD-TTS data pipeline.

This runner mirrors ``external_repro/official/hdtts/experiments/run_realworld.py``
for METR-LA.  The only intentional changes are (1) loading a pre-generated
observed-mask file and (2) selecting either the official HD-TTS-AMP model or
the MR-MSGF subclass.  It does not edit the upstream repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import sys

import numpy as np
import torch
import tsl.datasets as tsl_datasets
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from scipy.sparse import csr_matrix
from tsl.data import SpatioTemporalDataModule, SpatioTemporalDataset
from tsl.data.preprocessing import scalers
from tsl.metrics import torch_metrics

from ..models.hdtts_official_reliable import OfficialMRMSGFHDTTSModel


ROOT = pathlib.Path(__file__).resolve().parents[3]
OFFICIAL_HDTTS = ROOT / "external_repro" / "official" / "hdtts"
if str(OFFICIAL_HDTTS) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_HDTTS))

from lib.nn.models.hdtts_model import HDTTSModel  # noqa: E402
from lib.nn.predictors import MultiScalePredictor  # noqa: E402


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
    seed_everything(seed, workers=True)


def load_observed_mask(path: pathlib.Path, shape: tuple[int, int, int]) -> np.ndarray:
    mask = np.load(path)
    if mask.ndim == 2:
        mask = mask[..., None]
    if mask.shape != shape:
        raise ValueError(f"Mask {path} has shape {mask.shape}; expected {shape}.")
    if not np.isfinite(mask).all() or not np.isin(mask, [0, 1]).all():
        raise ValueError(f"Mask {path} must be a finite binary observed mask.")
    return mask.astype(bool, copy=False)


def make_datamodule(mask_path: pathlib.Path, batch_size: int):
    """Reproduce the official METR-LA preprocessing and split construction."""
    dataset = tsl_datasets.MetrLA(impute_zeros=True)
    adjacency = dataset.get_connectivity(method="distance", threshold=0.1,
                                         include_self=False, layout="csr")
    original_mask = dataset.get_mask().copy()
    observed_mask = load_observed_mask(mask_path, original_mask.shape)
    mask = np.logical_and(original_mask, observed_mask)
    dataset.set_mask(mask)

    u = [dataset.datetime_encoded("day").values,
         dataset.datetime_onehot("weekday").values]
    u = np.concatenate([np.repeat(item[:, None], dataset.n_nodes, 1)
                        if item.ndim < 3 else item for item in u], axis=-1)

    data = dataset.dataframe()
    masked_data = data.where(mask.reshape(mask.shape[0], -1), np.nan)
    data = masked_data.ffill().bfill()
    data.fillna(0, inplace=True)

    torch_dataset = SpatioTemporalDataset(
        target=data,
        mask=mask,
        covariates=dict(u=u),
        connectivity=adjacency,
        horizon=12,
        window=24,
        stride=1,
    )
    torch_dataset.update_input_map(input_mask=["mask"])
    transform = dict(target=scalers.StandardScaler(axis=(0, 1)))
    datamodule = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=transform,
        splitter=dataset.get_splitter(val_len=0.1, test_len=0.2),
        mask_scaling=True,
        batch_size=batch_size,
        workers=0,
    )
    datamodule.setup()
    return datamodule, torch_dataset, adjacency, original_mask, dataset, data, mask


def build_predictor(args: argparse.Namespace, torch_dataset: SpatioTemporalDataset,
                    adjacency: csr_matrix) -> MultiScalePredictor:
    model_cls = HDTTSModel if args.model == "official_base" else OfficialMRMSGFHDTTSModel
    model_kwargs = dict(
        n_nodes=torch_dataset.n_nodes,
        input_size=torch_dataset.n_channels,
        mask_size=torch_dataset.n_channels,
        exog_size=torch_dataset.input_map.u.shape[-1],
        horizon=torch_dataset.horizon,
        hidden_size=64,
        dilation=3,
        rnn_layers=4,
        cell="gru",
        mp_method=["anisoconv", "propconv"],
        mp_kernel_size=1,
        pooling_layers=3,
        mp_stage="both",
        recursive_lifting=True,
        fully_connected_readout=True,
        multi_step_scores=True,
        dropout=0.0,
        activation="elu",
    )
    if args.model == "mrmsgf":
        model_kwargs.update(
            physical_adjacency=torch.as_tensor(adjacency.toarray(), dtype=torch.float32),
            variant=args.mr_variant,
            graph_topk=args.graph_topk,
            reliability_beta=args.reliability_beta,
            disable_overlap_reliability=args.disable_overlap_reliability,
            disable_readout_reliability=args.disable_readout_reliability,
        )
    else:
        model_cls.filter_model_args_(model_kwargs)

    return MultiScalePredictor(
        model_class=model_cls,
        model_kwargs=model_kwargs,
        optim_class=torch.optim.AdamW,
        optim_kwargs=dict(lr=1e-3, weight_decay=0.01),
        loss_fn=torch_metrics.MaskedMAE(),
        metrics=dict(mae=torch_metrics.MaskedMAE(),
                     mse=torch_metrics.MaskedMSE(),
                     mre=torch_metrics.MaskedMRE()),
        scheduler_class=torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_kwargs=dict(monitor="val_loss", mode="min", factor=0.5,
                              patience=10, threshold=1e-4, min_lr=1e-6),
        scale_target=False,
        whiten_prob=None,
        imputation_loss_fn=None,
        imputation_loss_weight=1.0,
        imputation_warm_up=6,
    )


def as_floats(items: list[dict]) -> dict[str, float]:
    return {key: float(value) for key, value in (items[0] if items else {}).items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["official_base", "mrmsgf"], required=True)
    parser.add_argument("--mask_path", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--train_batches", type=int, default=300)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--graph_topk", type=int, default=8)
    parser.add_argument("--reliability_beta", type=float, default=1.0)
    parser.add_argument("--mr_variant", choices=sorted(OfficialMRMSGFHDTTSModel.VALID_VARIANTS), default="full")
    parser.add_argument("--disable_overlap_reliability", action="store_true")
    parser.add_argument("--disable_readout_reliability", action="store_true")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_path = pathlib.Path(args.mask_path).resolve()
    set_seed(args.seed)
    datamodule, torch_dataset, adjacency, original_mask, dataset, data, mask = make_datamodule(mask_path, args.batch_size)
    predictor = build_predictor(args, torch_dataset, adjacency)

    run_dir = out_dir / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(dirpath=run_dir, save_top_k=1, monitor="val_mae", mode="min")
    trainer = Trainer(
        max_epochs=args.epochs,
        limit_train_batches=args.train_batches,
        default_root_dir=run_dir,
        logger=TensorBoardLogger(save_dir=run_dir),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        gradient_clip_val=5,
        callbacks=[EarlyStopping(monitor="val_mae", patience=args.patience, mode="min"), checkpoint],
        enable_progress_bar=False,
    )
    trainer.fit(predictor, train_dataloaders=datamodule.train_dataloader(),
                val_dataloaders=datamodule.val_dataloader())
    # TSL's load_model compares every model kwarg with ``==``. MR-MSGF has a
    # fixed physical-adjacency tensor kwarg, for which that scalar comparison
    # is undefined; load the exact checkpoint state instead.
    checkpoint_state = torch.load(checkpoint.best_model_path, map_location="cpu")
    predictor.load_state_dict(checkpoint_state["state_dict"])
    predictor.freeze()
    masked_test = as_floats(trainer.test(predictor, dataloaders=datamodule.test_dataloader()))

    # This is the upstream runner's second test pass on the original target mask.
    torch_dataset.set_mask(original_mask)
    torch_dataset.set_data(dataset.numpy())
    torch_dataset.add_covariate("x", data, "t n f", add_to_input_map=True, preprocess=True)
    torch_dataset.add_scaler("x", torch_dataset.scalers["target"])
    torch_dataset.add_covariate("input_mask", mask, "t n f", add_to_input_map=True)
    from torchmetrics import MetricCollection
    predictor.test_metrics = MetricCollection(
        metrics={name: predictor._check_metric(metric)
                 for name, metric in dict(mae=torch_metrics.MaskedMAE(),
                                          mse=torch_metrics.MaskedMSE(),
                                          mre=torch_metrics.MaskedMRE()).items()},
        prefix="test_", postfix="_unmasked",
    )
    unmasked_test = as_floats(trainer.test(predictor, dataloaders=datamodule.test_dataloader()))

    row = dict(
        tag=args.tag,
        model=args.model,
        mr_variant=args.mr_variant if args.model == "mrmsgf" else None,
        disable_overlap_reliability=bool(args.disable_overlap_reliability) if args.model == "mrmsgf" else None,
        disable_readout_reliability=bool(args.disable_readout_reliability) if args.model == "mrmsgf" else None,
        seed=args.seed,
        epochs=args.epochs,
        train_batches=args.train_batches,
        mask_path=str(mask_path),
        mask_sha256=sha256(mask_path),
        artificial_missing_rate=float(1.0 - np.load(mask_path).mean()),
        official_hdtts_commit="46a8717db3802e1684e991ba0a7c0bfe43d22535",
        data_sha256=sha256(ROOT / "dataset" / "METR-LA" / "metr-la.h5"),
        model_hparams=dict(hidden_size=64, dilation=3, rnn_layers=4,
                           pooling_layers=3, mp_method=["anisoconv", "propconv"],
                           fully_connected_readout=True, multi_step_scores=True),
        masked_test=masked_test,
        unmasked_test=unmasked_test,
        best_val_mae=float(checkpoint.best_model_score.cpu()),
    )
    result_path = out_dir / f"{args.tag}.json"
    result_path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
