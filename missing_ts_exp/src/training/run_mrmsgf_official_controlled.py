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
import subprocess
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

from ..models.hdtts_official_reliable import (
    OfficialFeatureOnlyHDTTSModel,
    OfficialMRMSGFHDTTSModel,
)


ROOT = pathlib.Path(__file__).resolve().parents[3]
OFFICIAL_HDTTS = ROOT / "external_repro" / "official" / "hdtts"
if str(OFFICIAL_HDTTS) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_HDTTS))

from lib.nn.models.hdtts_model import HDTTSModel  # noqa: E402
from lib.nn.models.baselines.grin_pred_model import GRINPredictionModel  # noqa: E402
from lib.nn.predictors import MultiScalePredictor  # noqa: E402


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else f"git-command-failed:{result.stderr.strip()}"


def source_manifest() -> dict[str, object]:
    """Capture the exact code identity that started this training process."""
    controlled = (
        ROOT / "missing_ts_exp/src/models/hdtts_official_reliable.py",
        ROOT / "missing_ts_exp/src/training/run_mrmsgf_official_controlled.py",
    )
    files = {str(path.relative_to(ROOT)): sha256(path) for path in controlled}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "head": git_output("rev-parse", "HEAD"),
        "status_porcelain": git_output("status", "--porcelain=v1"),
        "controlled_files_sha256": files,
        "controlled_source_sha256": digest,
    }


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


def make_datamodule(mask_path: pathlib.Path, batch_size: int, include_window_metadata: bool = False):
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
    if include_window_metadata:
        # These covariates make a sample's permutation independent of loader
        # order. The first value of each input window is its absolute start.
        time_index = np.broadcast_to(
            np.arange(mask.shape[0], dtype=np.int64)[:, None, None], mask.shape
        ).copy()
        split_id = np.zeros(mask.shape, dtype=np.int64)
        for value, indices in enumerate((datamodule.splitter.train_idxs,
                                         datamodule.splitter.val_idxs,
                                         datamodule.splitter.test_idxs)):
            split_id[indices, :, :] = value
        torch_dataset.add_covariate("window_index", time_index, "t n f", add_to_input_map=True)
        torch_dataset.add_covariate("split_id", split_id, "t n f", add_to_input_map=True)
    return datamodule, torch_dataset, adjacency, original_mask, dataset, data, mask


def build_predictor(args: argparse.Namespace, torch_dataset: SpatioTemporalDataset,
                    adjacency: csr_matrix) -> MultiScalePredictor:
    if args.model == "grin":
        model_cls = GRINPredictionModel
        model_kwargs = dict(
            n_nodes=torch_dataset.n_nodes,
            input_size=torch_dataset.n_channels,
            exog_size=torch_dataset.input_map.u.shape[-1],
            horizon=torch_dataset.horizon,
            hidden_size=64,
            n_layers=1,
            embedding_size=None,
            kernel_size=2,
            decoder_order=1,
            layer_norm=False,
            activation="relu",
            dropout=0.0,
        )
    else:
        if args.model == "official_base":
            model_cls = HDTTSModel
        elif args.model == "official_base_feature_only":
            model_cls = OfficialFeatureOnlyHDTTSModel
        else:
            model_cls = OfficialMRMSGFHDTTSModel
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
    if args.model in {"mrmsgf", "official_base_feature_only"}:
        region_assignment = np.load(args.region_assignment_path)
        model_kwargs.update(
            physical_adjacency=torch.as_tensor(adjacency.toarray(), dtype=torch.float32),
            region_assignment=torch.as_tensor(region_assignment, dtype=torch.float32),
        )
        if args.model == "mrmsgf":
            model_kwargs.update(
                variant=args.mr_variant,
                graph_topk=args.graph_topk,
                reliability_beta=args.reliability_beta,
                q_mode=args.q_mode,
                mask_seed=args.mask_seed,
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
        imputation_loss_fn=torch_metrics.MaskedMAE() if args.model == "grin" else None,
        imputation_loss_weight=1.0,
        imputation_warm_up=6,
    )


def as_floats(items: list[dict]) -> dict[str, float]:
    return {key: float(value) for key, value in (items[0] if items else {}).items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["official_base", "official_base_feature_only", "grin", "mrmsgf"], required=True)
    parser.add_argument("--protocol_id", default="ad_hoc")
    parser.add_argument("--mask_path", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--train_batches", type=int, default=300)
    parser.add_argument("--eval_batches", type=int,
                        help="Limit validation/test batches for a non-formal smoke run only.")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--graph_topk", type=int, default=8)
    parser.add_argument("--reliability_beta", type=float, default=1.0)
    parser.add_argument("--q_mode", choices=sorted(OfficialMRMSGFHDTTSModel.VALID_Q_MODES), default="observed")
    parser.add_argument("--mask_seed", type=int)
    parser.add_argument("--mr_variant", choices=sorted(OfficialMRMSGFHDTTSModel.VALID_VARIANTS), default="full")
    parser.add_argument("--disable_overlap_reliability", action="store_true")
    parser.add_argument("--disable_readout_reliability", action="store_true")
    parser.add_argument(
        "--region_assignment_path",
        default=str(ROOT / "missing_ts_exp/resources/r0905_blockt/regions/METR-LA/physical_connected_max8_assignment.npy"),
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_path = pathlib.Path(args.mask_path).resolve()
    args.region_assignment_path = str(pathlib.Path(args.region_assignment_path).resolve())
    if args.model == "mrmsgf" and args.q_mode == "permuted_pairwise" and args.mask_seed is None:
        parser.error("--mask_seed is required with --q_mode permuted_pairwise")
    set_seed(args.seed)
    datamodule, torch_dataset, adjacency, original_mask, dataset, data, mask = make_datamodule(
        mask_path, args.batch_size,
        include_window_metadata=args.model in {"mrmsgf", "official_base_feature_only"},
    )
    predictor = build_predictor(args, torch_dataset, adjacency)

    run_dir = out_dir / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(dirpath=run_dir, save_top_k=1, monitor="val_mae", mode="min")
    trainer_kwargs = dict(
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
    if args.eval_batches is not None:
        trainer_kwargs.update(limit_val_batches=args.eval_batches, limit_test_batches=args.eval_batches)
    trainer = Trainer(**trainer_kwargs)
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
        protocol_id=args.protocol_id,
        tag=args.tag,
        model=args.model,
        mr_variant=args.mr_variant if args.model == "mrmsgf" else None,
        q_mode=args.q_mode if args.model == "mrmsgf" else "not_applicable",
        mask_seed=args.mask_seed,
        q_permutation=(dict(
            algorithm="blake2b-64/mrmsgf-qperm-v1",
            transform="Q_perm=P Q P^T; diag(Q_perm)=diag(Q)",
            seed_components=["mask_seed", "split_id", "absolute_window_index"],
            window_metadata_input=True,
            diagonal_restored=True,
            off_diagonal_multiset_preserved=True,
        ) if args.model == "mrmsgf" and args.q_mode == "permuted_pairwise" else None),
        feature_only_no_dynamic_graph=True if args.model == "official_base_feature_only" else None,
        zero_observation_physical_fallback=args.model == "mrmsgf",
        dispersion_numerical_protection=args.model == "mrmsgf",
        window_metadata=dict(
            input_enabled=args.model in {"mrmsgf", "official_base_feature_only"},
            input_fields=["window_index", "split_id"],
            consumed_for_permutation=args.model == "mrmsgf" and args.q_mode == "permuted_pairwise",
        ),
        disable_overlap_reliability=bool(args.disable_overlap_reliability) if args.model == "mrmsgf" else False,
        disable_readout_reliability=bool(args.disable_readout_reliability) if args.model == "mrmsgf" else False,
        seed=args.seed,
        epochs=args.epochs,
        train_batches=args.train_batches,
        eval_batches=args.eval_batches,
        patience=args.patience,
        batch_size=args.batch_size,
        graph_topk=args.graph_topk,
        reliability_beta=args.reliability_beta,
        mask_path=str(mask_path),
        mask_sha256=sha256(mask_path),
        artificial_missing_rate=float(1.0 - np.load(mask_path).mean()),
        official_hdtts_commit="46a8717db3802e1684e991ba0a7c0bfe43d22535",
        data_sha256=sha256(ROOT / "dataset" / "METR-LA" / "metr-la.h5"),
        region_assignment_path=args.region_assignment_path if args.model in {"mrmsgf", "official_base_feature_only"} else None,
        region_assignment_sha256=(sha256(pathlib.Path(args.region_assignment_path))
                                  if args.model in {"mrmsgf", "official_base_feature_only"} else None),
        parameter_count=sum(parameter.numel() for parameter in predictor.model.parameters()),
        best_checkpoint=checkpoint.best_model_path,
        best_epoch=int(checkpoint_state.get("epoch", -1)),
        optimizer=dict(name="AdamW", lr=1e-3, weight_decay=0.01),
        source_manifest=source_manifest(),
        model_hparams=(dict(hidden_size=64, n_layers=1, kernel_size=2, decoder_order=1,
                            imputation_loss=True) if args.model == "grin" else
                       dict(hidden_size=64, dilation=3, rnn_layers=4,
                            pooling_layers=3, mp_method=["anisoconv", "propconv"],
                            fully_connected_readout=True, multi_step_scores=True)),
        masked_test=masked_test,
        unmasked_test=unmasked_test,
        best_val_mae=float(checkpoint.best_model_score.cpu()),
    )
    result_path = out_dir / f"{args.tag}.json"
    result_path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
