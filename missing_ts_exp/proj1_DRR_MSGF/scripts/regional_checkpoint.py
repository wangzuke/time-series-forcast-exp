"""Restore the recorded model and evaluation inputs for regional analyses."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from ..src.training.run_mrmsgf_official_controlled import (
    build_predictor,
    make_datamodule,
    restore_evaluation_targets,
    sha256,
)


def load_checkpoint(result: Path, batch_size: int, device: str):
    row = json.loads(result.read_text())
    mask = Path(row["mask_path"])
    if sha256(mask) != row["mask_sha256"]:
        raise ValueError(f"Mask hash does not match {result}.")
    if row.get("region_assignment_sha256"):
        if sha256(Path(row["region_assignment_path"])) != row["region_assignment_sha256"]:
            raise ValueError(f"Region assignment hash does not match {result}.")
    dm, windows, adjacency, original, dataset, data, input_mask = make_datamodule(
        mask, batch_size, include_window_metadata=True,
    )
    predictor = build_predictor(SimpleNamespace(**row), windows, adjacency)
    path = Path(row["best_checkpoint"])
    if not path.is_file():
        path = result.parent / row["tag"] / path.name
    predictor.load_state_dict(torch.load(path, map_location="cpu")["state_dict"])
    predictor.to(device).eval()
    restore_evaluation_targets(windows, dataset, original, data, input_mask)
    return row, dm, predictor
