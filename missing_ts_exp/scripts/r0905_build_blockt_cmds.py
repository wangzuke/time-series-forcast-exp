"""Build command files for the R0905 Point/Block-T experiment."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESOURCE_DIR = ROOT / "resources" / "r0905_blockt"
PYTHON = "/media/data1/wangzk/miniconda3/envs/hd-tts/bin/python"


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def rate_label(rate: str) -> str:
    return f"{int(round(float(rate) * 100)):02d}"


def region_assignment_path(resource_dir: Path, dataset: str, max_region_size: int = 8) -> Path:
    path = resource_dir / "regions" / dataset / f"physical_connected_max{max_region_size}_assignment.npy"
    if not path.is_file():
        raise FileNotFoundError(f"Missing verified region assignment: {path}. Run r0906_prepare_mrmsgf_regions.py first.")
    return path


def make_cmd(row: dict[str, str], model: str, out_dir: Path, epochs: int, batch_size: int,
             resource_dir: Path,
             max_train_batches: int = -1, max_eval_batches: int = -1) -> str:
    mode = row["mode"]
    missing_type = "none" if mode == "original" else mode
    missing_rate = float(row["target_missing_rate"])
    tag = (
        f"{model}_{row['dataset']}_{mode}_r{rate_label(row['target_missing_rate'])}_"
        f"seed{row['seed']}"
    )
    parts = [
        PYTHON,
        "-m", "missing_ts_exp.src.training.run_official_hdtts_local",
        "--model", model,
        "--dataset", row["dataset"],
        "--missing_type", missing_type,
        "--missing_rate", f"{missing_rate:.4f}",
        "--mask_path", row["mask_path"],
        "--seed", row["seed"],
        "--epochs", str(epochs),
        "--patience", "20" if epochs > 5 else "2",
        "--batch_size", str(batch_size),
        "--num_workers", "0",
        "--out_dir", str(out_dir),
        "--tag", tag,
    ]
    if model == "mrmsgf":
        parts.extend(["--region_assignment_path", str(region_assignment_path(resource_dir, row["dataset"]))])
    if max_train_batches >= 0:
        parts.extend(["--max_train_batches", str(max_train_batches)])
    if max_eval_batches >= 0:
        parts.extend(["--max_eval_batches", str(max_eval_batches)])
    return " ".join(parts)


def build_smoke(rows: list[dict[str, str]], out_dir: Path, resource_dir: Path) -> list[str]:
    smoke_rows = [
        r for r in rows
        if r["dataset"] == "METR-LA"
        and r["mode"] == "block_t"
        and abs(float(r["target_missing_rate"]) - 0.50) < 1e-6
        and r["seed"] == "2024"
    ]
    if len(smoke_rows) != 1:
        raise RuntimeError(f"expected one METR-LA block_t 50 seed2024 row, got {len(smoke_rows)}")
    row = smoke_rows[0]
    return [
        make_cmd(row, "official_base", out_dir / "smoke", epochs=1, batch_size=16, resource_dir=resource_dir, max_train_batches=2, max_eval_batches=1),
        make_cmd(row, "mrmsgf", out_dir / "smoke", epochs=1, batch_size=16, resource_dir=resource_dir, max_train_batches=2, max_eval_batches=1),
    ]


def build_main(rows: list[dict[str, str]], out_dir: Path, epochs: int, batch_size: int, resource_dir: Path) -> list[str]:
    selected = []
    for row in rows:
        dataset = row["dataset"]
        mode = row["mode"]
        rate = float(row["target_missing_rate"])
        if dataset == "METR-LA" and mode in {"point", "block_t"} and rate in {0.30, 0.50, 0.70}:
            selected.append(row)
        elif dataset == "AQI36" and mode in {"original", "point"}:
            selected.append(row)
    cmds: list[str] = []
    for row in selected:
        for model in ("official_base", "mrmsgf"):
            cmds.append(make_cmd(row, model, out_dir / "main", epochs=epochs, batch_size=batch_size, resource_dir=resource_dir))
    return cmds


def build_stage1_core(rows: list[dict[str, str]], out_dir: Path, epochs: int, batch_size: int, resource_dir: Path) -> list[str]:
    selected = [
        r for r in rows
        if r["dataset"] == "METR-LA"
        and r["mode"] == "block_t"
        and r["seed"] == "2024"
        and float(r["target_missing_rate"]) in {0.50, 0.70}
    ]
    cmds: list[str] = []
    for row in selected:
        for model in ("official_base", "mrmsgf"):
            cmds.append(make_cmd(row, model, out_dir / "stage1_core", epochs=epochs, batch_size=batch_size, resource_dir=resource_dir))
    return cmds


def build_stage2_blockt_all(rows: list[dict[str, str]], out_dir: Path, epochs: int, batch_size: int, resource_dir: Path) -> list[str]:
    selected = [
        r for r in rows
        if r["dataset"] == "METR-LA"
        and r["mode"] == "block_t"
        and float(r["target_missing_rate"]) in {0.30, 0.50, 0.70}
    ]
    cmds: list[str] = []
    for row in selected:
        for model in ("official_base", "mrmsgf"):
            cmds.append(make_cmd(row, model, out_dir / "stage2_blockt_all", epochs=epochs, batch_size=batch_size, resource_dir=resource_dir))
    return cmds


def build_stage3_aux(rows: list[dict[str, str]], out_dir: Path, epochs: int, batch_size: int, resource_dir: Path) -> list[str]:
    selected = []
    for row in rows:
        dataset = row["dataset"]
        mode = row["mode"]
        rate = float(row["target_missing_rate"])
        if dataset == "METR-LA" and mode == "point" and rate in {0.30, 0.50, 0.70}:
            selected.append(row)
        elif dataset == "AQI36" and mode in {"original", "point"}:
            selected.append(row)
    cmds: list[str] = []
    for row in selected:
        for model in ("official_base", "mrmsgf"):
            cmds.append(make_cmd(row, model, out_dir / "stage3_aux", epochs=epochs, batch_size=batch_size, resource_dir=resource_dir))
    return cmds


def build_ablation(rows: list[dict[str, str]], out_dir: Path, epochs: int, batch_size: int, resource_dir: Path) -> list[str]:
    selected = [
        r for r in rows
        if r["dataset"] == "METR-LA"
        and r["mode"] == "block_t"
        and float(r["target_missing_rate"]) in {0.50, 0.70}
    ]
    variants = [
        ("mrmsgf_phy_only", ["--mrmsgf_variant", "phy_only"]),
        ("mrmsgf_obs_only", ["--mrmsgf_variant", "obs_only"]),
        ("mrmsgf_fine", ["--mrmsgf_variant", "fine"]),
        ("mrmsgf_coarse_no_gate", ["--mrmsgf_variant", "coarse_no_gate"]),
        ("mrmsgf_full", ["--mrmsgf_variant", "full"]),
    ]
    cmds: list[str] = []
    for row in selected:
        for name, extra in variants:
            cmd = make_cmd(row, "mrmsgf", out_dir / "ablation", epochs=epochs, batch_size=batch_size, resource_dir=resource_dir)
            cmd = cmd.replace(
                f"mrmsgf_{row['dataset']}",
                f"{name}_{row['dataset']}",
            )
            cmds.append(cmd + " " + " ".join(extra))
    return cmds


def write_cmds(path: Path, cmds: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(cmds) + "\n")
    print(f"wrote {len(cmds)} commands: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource_dir", default=str(DEFAULT_RESOURCE_DIR))
    parser.add_argument("--out_dir", default=str(ROOT / "results" / "r0906_mrmsgf"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    resource_dir = Path(args.resource_dir)
    rows = load_manifest(resource_dir / "manifest.csv")
    out_dir = Path(args.out_dir)
    script_dir = ROOT / "scripts"
    write_cmds(script_dir / "r0906_mrmsgf_smoke_cmds.txt", build_smoke(rows, out_dir, resource_dir))
    write_cmds(script_dir / "r0906_mrmsgf_stage1_core_cmds.txt", build_stage1_core(rows, out_dir, args.epochs, args.batch_size, resource_dir))
    write_cmds(script_dir / "r0906_mrmsgf_stage2_blockt_all_cmds.txt", build_stage2_blockt_all(rows, out_dir, args.epochs, args.batch_size, resource_dir))
    write_cmds(script_dir / "r0906_mrmsgf_stage3_aux_cmds.txt", build_stage3_aux(rows, out_dir, args.epochs, args.batch_size, resource_dir))
    write_cmds(script_dir / "r0906_mrmsgf_main_cmds.txt", build_main(rows, out_dir, args.epochs, args.batch_size, resource_dir))
    write_cmds(script_dir / "r0906_mrmsgf_ablation_cmds.txt", build_ablation(rows, out_dir, args.epochs, args.batch_size, resource_dir))


if __name__ == "__main__":
    main()
