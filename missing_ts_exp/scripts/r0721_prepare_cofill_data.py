#!/usr/bin/env python3
"""Prepare local data assets required by the CoFILL official code.

The official CoFILL repository expects files under:

  external_repro/CoFILL/data/metr_la/
  external_repro/CoFILL/data/pems_bay/

This helper keeps the canonical 0721 data source in `dataset/` and creates the
minimal compatibility assets CoFILL needs: HDF5 links, train mean/std pickle,
and distance matrices used by `generate_adj.py`.
"""

from __future__ import annotations

import csv
import os
import pathlib
import pickle

import numpy as np
import pandas as pd


ROOT = pathlib.Path("/data/wangzuke/time-series-forecast-exp")
COFILL = ROOT / "external_repro" / "CoFILL"
DATASET = ROOT / "dataset"
GRAPH = DATASET / "_archives" / "dcrnn_sensor_graph"


DATASETS = {
    "Metr": {
        "h5": DATASET / "metr_la" / "metr_la.h5",
        "cofill_dir": COFILL / "data" / "metr_la",
        "cofill_h5": "metr_la.h5",
        "meanstd": "metr_meanstd.pk",
        "dist": "metr_la_dist.npy",
        "dist_csv": GRAPH / "distances_la_2012.csv",
        "ids": GRAPH / "graph_sensor_ids.txt",
    },
    "PEMS": {
        "h5": DATASET / "pems_bay" / "pems_bay.h5",
        "cofill_dir": COFILL / "data" / "pems_bay",
        "cofill_h5": "pems_bay.h5",
        "meanstd": "pems_meanstd.pk",
        "dist": "pems_bay_dist.npy",
        "dist_csv": GRAPH / "distances_bay_2017.csv",
        "ids": None,
    },
}


def read_h5(path: pathlib.Path) -> pd.DataFrame:
    obj = pd.read_hdf(path)
    if not isinstance(obj, pd.DataFrame):
        raise TypeError(f"{path} did not load as a DataFrame")
    return obj


def safe_link(src: pathlib.Path, dst: pathlib.Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and pathlib.Path(os.readlink(dst)) == src:
            return
        dst.unlink()
    os.symlink(src, dst)


def write_meanstd(h5: pathlib.Path, out: pathlib.Path) -> None:
    df = read_h5(h5)
    data_len = len(df)
    train_data = df.iloc[: int(data_len * 0.7)].fillna(0).values.astype(np.float32)
    mean = np.mean(train_data, axis=0).astype(np.float32)
    std = np.std(train_data, axis=0).astype(np.float32)
    std[std == 0] = 1.0
    with out.open("wb") as f:
        pickle.dump((mean, std), f)


def ids_from_h5(h5: pathlib.Path) -> list[str]:
    df = read_h5(h5)
    return [str(c) for c in df.columns]


def ids_from_file(path: pathlib.Path) -> list[str]:
    text = path.read_text().strip()
    return [x.strip() for x in text.replace("\n", ",").split(",") if x.strip()]


def load_distance_rows(path: pathlib.Path) -> list[tuple[str, str, float]]:
    rows: list[tuple[str, str, float]] = []
    with path.open(newline="") as f:
        sample = f.read(512)
        f.seek(0)
        has_header = "from" in sample.lower() and "to" in sample.lower()
        if has_header:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append((str(r["from"]), str(r["to"]), float(r["cost"])))
        else:
            reader = csv.reader(f)
            for r in reader:
                if len(r) >= 3:
                    rows.append((str(r[0]), str(r[1]), float(r[2])))
    return rows


def write_distance_matrix(dataset_name: str, info: dict[str, pathlib.Path | str | None], out: pathlib.Path) -> None:
    if info["ids"] is not None:
        ids = ids_from_file(pathlib.Path(info["ids"]))
    else:
        ids = ids_from_h5(pathlib.Path(info["h5"]))
    idx = {sensor_id: i for i, sensor_id in enumerate(ids)}
    n = len(ids)
    dist = np.full((n, n), np.inf, dtype=np.float32)
    np.fill_diagonal(dist, 0.0)
    used = 0
    for src, dst, cost in load_distance_rows(pathlib.Path(info["dist_csv"])):
        if src in idx and dst in idx:
            dist[idx[src], idx[dst]] = float(cost)
            used += 1
    if used == 0:
        raise RuntimeError(f"{dataset_name}: no distance rows matched sensor ids")
    np.save(out, dist)


def main() -> int:
    if not COFILL.exists():
        raise FileNotFoundError(f"CoFILL repository missing: {COFILL}")

    for dataset_name, info in DATASETS.items():
        h5 = pathlib.Path(info["h5"])
        out_dir = pathlib.Path(info["cofill_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        safe_link(h5, out_dir / str(info["cofill_h5"]))
        write_meanstd(h5, out_dir / str(info["meanstd"]))
        write_distance_matrix(dataset_name, info, out_dir / str(info["dist"]))

        print(
            f"[cofill_data] {dataset_name} h5={out_dir / str(info['cofill_h5'])} "
            f"meanstd={out_dir / str(info['meanstd'])} dist={out_dir / str(info['dist'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
