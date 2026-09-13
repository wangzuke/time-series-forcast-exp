#!/usr/bin/env python3
"""Build the paired METR-LA Point main-table experiment matrix."""
from pathlib import Path


ROOT = Path("/media/data1/wangzk/time-series-forcast-exp")
OUT = ROOT / "missing_ts_exp/scripts/r0910_stage_a2_cmds.txt"
MASKS = ROOT / "missing_ts_exp/resources/r0905_blockt/masks/METR-LA"
RESULT = ROOT / "missing_ts_exp/results/r0910_evidence_closure/stage_a2_point"
PYTHON = "/media/data1/wangzk/miniconda3/envs/hd-tts/bin/python"
HDTTS_ROOT = ROOT / "external_repro/official/hdtts"


def controlled(model: str, rate: int, seed: int) -> str:
    mask = MASKS / f"point_r{rate}_seed{seed}_observed.npy"
    tag = f"{model}_METR-LA_point_r{rate}_seed{seed}"
    return (
        f"{PYTHON} -m missing_ts_exp.src.training.run_mrmsgf_official_controlled "
        f"--model {model} --mr_variant full --mask_path {mask} --seed {seed} "
        f"--epochs 100 --train_batches 300 --patience 20 --batch_size 32 "
        f"--out_dir {RESULT} --tag {tag}"
    )


def bitgraph(rate: int, seed: int) -> str:
    mask = MASKS / f"point_r{rate}_seed{seed}_observed.npy"
    tag = f"bitgraph_unified_METR-LA_point_r{rate}_seed{seed}"
    return (
        f"PYTHONPATH={HDTTS_ROOT} {PYTHON} -m "
        "missing_ts_exp.src.training.run_bitgraph_official_unified "
        f"--mask_path {mask} --seed {seed} --epochs 200 --batch_size 32 "
        f"--out_dir {RESULT} --tag {tag}"
    )


def main() -> None:
    commands = []
    for rate in (30, 50):
        for seed in (2024, 2025, 2026, 2027, 2028):
            commands.extend((controlled("official_base", rate, seed), controlled("mrmsgf", rate, seed), bitgraph(rate, seed)))
    OUT.write_text("\n".join(commands) + "\n")
    print(f"wrote {len(commands)} commands to {OUT}")


if __name__ == "__main__":
    main()
