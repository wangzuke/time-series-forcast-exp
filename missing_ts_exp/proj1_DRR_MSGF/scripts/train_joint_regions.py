"""Train or evaluate a cached joint-region ablation under the P1 protocol."""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from ..src.data.joint_regions import membership_changes
from ..src.training.run_mrmsgf_official_controlled import (
    PROJECT_ROOT, build_predictor, make_datamodule, restore_evaluation_targets, set_seed, sha256, source_manifest,
)


@torch.inference_mode()
def audit_graph(predictor, loader, path: Path) -> float:
    archive = np.load(path)
    inputs = next(iter(loader)).to("cuda").input
    model = predictor.model
    ids = inputs["window_index"].reshape(inputs["x"].size(0), -1)[:, 0].cpu().numpy()
    expected = {int(key): graph for key, graph in zip(archive["starts"], archive["graph"])}
    graph = model._graph_inputs(inputs["x"], inputs["input_mask"], inputs["u"])[-2]
    reference = graph.new_tensor(np.stack([expected[int(idx)] for idx in ids]))
    if not torch.allclose(graph, reference, atol=1e-6, rtol=1e-5):
        raise ValueError("The training graph differs from the graph used to generate the partition cache.")
    altered = inputs["x"].masked_fill(~inputs["input_mask"].bool(), 1000)
    rechecked = model._graph_inputs(altered, inputs["input_mask"], inputs["u"])[-2]
    if not torch.equal(graph, rechecked):
        raise ValueError("An unobserved input value changed the fused graph.")

    assignment = model._cached_assignment(inputs["x"], inputs["window_index"])
    padded = torch.nn.functional.pad(assignment, (0, 2))
    graph_padded = model._regional_adjacencies(padded, graph, graph, graph)["fuse_then_coarsen"]
    graph_regular = model._regional_adjacencies(assignment, graph, graph, graph)["fuse_then_coarsen"]
    rate = torch.ones_like(assignment[..., 0])
    latent = inputs["x"][:, 0].float().expand(-1, -1, model.encoder_time.hidden_size)
    for index in range(min(4, graph.size(0))):
        actual = model._coarse_propagate(latent, graph_padded, padded, rate, rate, rate, rate)[0][index]
        expected_lift = model._coarse_propagate(latent, graph_regular, assignment, rate, rate, rate, rate)[0][index]
        if not torch.allclose(actual, expected_lift, rtol=1e-5, atol=1e-6):
            raise ValueError("Regional padding changed the node-level propagated features.")
    return float((graph - reference).abs().max())


@torch.inference_mode()
def evaluate(predictor, loader, limit: int | None, dynamic=None, static=None) -> dict:
    totals: dict[str, list[float]] = {}
    for batch in itertools.islice(loader, limit):
        batch = batch.to("cuda")
        target, forecast, valid = predictor.predict_batch(batch, return_target=True)
        valid = valid.bool()
        error = forecast - target
        groups = {"all": valid}
        if dynamic is not None:
            ids = batch.input["window_index"].reshape(target.size(0), -1)[:, 0].cpu().tolist()
            changed = np.stack([membership_changes(dynamic[index], static) for index in ids])
            changed = torch.from_numpy(changed).to(valid.device)[:, None, :, None]
            groups.update(changed=valid & changed, unchanged=valid & ~changed)
        for group, mask in groups.items():
            selected = error.masked_select(mask)
            count = selected.numel()
            absolute, squared, n = totals.setdefault(group, [0.0, 0.0, 0])
            totals[group] = [absolute + float(selected.abs().sum()), squared + float(selected.square().sum()), n + count]
    return {name: dict(mae=absolute / n if n else None, rmse=math.sqrt(squared / n) if n else None, n=n)
            for name, (absolute, squared, n) in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--cap", type=int, required=True)
    parser.add_argument("--mode", choices=["static", "dynamic"], required=True)
    parser.add_argument("--stage", choices=["smoke", "pilot", "test"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    resources = json.loads((args.resources / "manifest.json").read_text())
    if resources["config_sha256"] != sha256(args.config):
        raise RuntimeError("Partition cache was generated with a different configuration.")
    if resources["model_source_sha256"] != sha256(PROJECT_ROOT / "src/models/hdtts_official_reliable.py"):
        raise RuntimeError("Partition cache was generated with different model code.")
    if args.stage == "test":
        if args.selection is None:
            parser.error("--selection is required for test")
        selection = json.loads(args.selection.read_text())
        if selection["cap"] != args.cap or selection["config_sha256"] != sha256(args.config):
            raise ValueError("Test requested for a capacity other than the locked selection.")
    parent = json.loads((PROJECT_ROOT / cfg["parent_result"]).read_text())
    teacher = Path(resources["teacher_checkpoint"])
    mask_path = Path(parent["mask_path"])
    labels = args.resources / f"{args.mode}_cap{args.cap}.npy"
    for path, expected in (
        (teacher, resources["teacher_sha256"]), (mask_path, resources["mask_sha256"]),
        (Path(parent["region_assignment_path"]), resources["physical_assignment_sha256"]),
        (args.resources / "graph_audit.npz", resources["audit_sha256"]),
        (labels, resources["labels_sha256"][labels.name]),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"Experiment resource hash mismatch: {path}")
    source = source_manifest((args.config.resolve(), Path(__file__).resolve(),
                              PROJECT_ROOT / "src/data/joint_regions.py"))
    set_seed(cfg["seed"])
    dm, windows, adjacency, original, dataset, data, input_mask = make_datamodule(
        mask_path, cfg["batch_size"], include_window_metadata=True)
    parent.update(mr_variant="full", fusion_mode="edgewise", regional_order="fuse_then_coarsen")
    predictor = build_predictor(SimpleNamespace(**parent), windows, adjacency)
    predictor.load_state_dict(torch.load(teacher, map_location="cpu")["state_dict"])
    predictor.model.freeze_joint_graph()
    predictor.model.set_window_regions(torch.from_numpy(np.load(labels)))
    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.mode}_cap{args.cap}"
    run = args.out / tag

    if args.stage == "test":
        saved = json.loads((args.selection.parent / "pilot" / f"{tag}.json").read_text())
        if saved["cache_sha256"] != sha256(labels) or saved["checkpoint_sha256"] != selection["checkpoints"][tag]:
            raise RuntimeError("The locked validation checkpoint or cache differs.")
        trained = Path(saved["best_checkpoint"])
        if sha256(trained) != saved["checkpoint_sha256"]:
            raise RuntimeError("The trained checkpoint changed after selection.")
        predictor.load_state_dict(torch.load(trained, map_location="cpu")["state_dict"])
        predictor.to("cuda").eval()
        restore_evaluation_targets(windows, dataset, original, data, input_mask)
        static = np.load(args.resources / f"static_cap{args.cap}.npy")
        dynamic = np.load(args.resources / f"dynamic_cap{args.cap}.npy")
        metrics = evaluate(predictor, dm.test_dataloader(), None, dynamic=dynamic, static=static)
        row = dict(tag=tag, cap=args.cap, mode=args.mode, test=metrics, best_checkpoint=str(trained),
                   checkpoint_sha256=saved["checkpoint_sha256"])
    else:
        predictor.to("cuda").eval()
        delta = audit_graph(predictor, dm.val_dataloader(), args.resources / "graph_audit.npz")
        if run.exists():
            raise FileExistsError(f"Existing checkpoints at {run}")
        smoke = args.stage == "smoke"
        limit = cfg["smoke_batches"] if smoke else None
        checkpoint = ModelCheckpoint(dirpath=run, save_top_k=1, monitor="val_mae", mode="min")
        trainer = Trainer(
            accelerator="gpu", devices=1, max_epochs=cfg["smoke_epochs"] if smoke else cfg["epochs"],
            limit_train_batches=limit or cfg["train_batches"], limit_val_batches=limit if smoke else 1.0,
            gradient_clip_val=5, callbacks=[
                checkpoint, EarlyStopping(monitor="val_mae", mode="min", patience=cfg["patience"])],
            logger=TensorBoardLogger(save_dir=run), enable_progress_bar=False,
        )
        trainer.fit(predictor, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())
        trained = Path(checkpoint.best_model_path)
        state = torch.load(trained, map_location="cpu")
        predictor.load_state_dict(state["state_dict"])
        predictor.to("cuda").eval()
        metrics = evaluate(predictor, dm.val_dataloader(), limit)
        if abs(metrics["all"]["mae"] - float(checkpoint.best_model_score)) > 5e-4:
            raise RuntimeError("Validation recomputation differs from the checkpoint selection metric.")
        row = dict(tag=tag, cap=args.cap, mode=args.mode, validation=metrics["all"],
                   best_epoch=int(state["epoch"]), best_checkpoint=str(trained),
                   checkpoint_sha256=sha256(trained), graph_audit_max_delta=delta)

    row.update(stage=args.stage, config_sha256=sha256(args.config), mask_sha256=resources["mask_sha256"],
               teacher_sha256=resources["teacher_sha256"], cache_sha256=sha256(labels), source=source)
    temporary = args.out / f"{tag}.json.tmp"
    temporary.write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.out / f"{tag}.json")
    print({k: v for k, v in row.items() if k != "source"}, flush=True)


if __name__ == "__main__":
    main()
