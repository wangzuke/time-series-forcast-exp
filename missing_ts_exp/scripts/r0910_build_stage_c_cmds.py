#!/usr/bin/env python3
"""Build formal AQI36 natural-missingness and Point experiment commands."""
from pathlib import Path


ROOT = Path("/media/data1/wangzk/time-series-forcast-exp")
OUT = ROOT / "missing_ts_exp/scripts/r0910_stage_c_cmds.txt"
MASKS = ROOT / "missing_ts_exp/resources/r0905_blockt/masks/AQI36"
REGION = ROOT / "missing_ts_exp/resources/r0905_blockt/regions/AQI36/physical_connected_max8_assignment.npy"
RESULT = ROOT / "missing_ts_exp/results/r0910_evidence_closure/stage_c_aqi36"
PYTHON = "/media/data1/wangzk/miniconda3/envs/hd-tts/bin/python"


def command(model: str, condition: str, rate: int, seed: int) -> str:
    if condition == "original":
        mask = MASKS / f"original_seed{seed}_observed.npy"
        missing_type, missing_rate, label = "none", "0.0", "original_r00"
    else:
        mask = MASKS / f"point_r{rate}_seed{seed}_observed.npy"
        missing_type, missing_rate, label = "point", f"{rate / 100:.2f}", f"point_r{rate}"
    tag = f"{model}_AQI36_{label}_seed{seed}"
    region = f" --region_assignment_path {REGION}" if model == "mrmsgf" else ""
    return (
        f"{PYTHON} -m missing_ts_exp.src.training.run_official_hdtts_local "
        f"--model {model} --dataset AQI36 --seq_len 24 --pred_len 12 "
        f"--missing_type {missing_type} --missing_rate {missing_rate} --mask_path {mask} "
        f"--seed {seed} --epochs 100 --patience 20 --batch_size 64 --num_workers 0 "
        f"--hidden 64 --rnn_layers 3 --pooling_layers 3 --dilation 4 "
        f"--mp_method anisoconv,anisoconv --train_loss mae{region} "
        f"--out_dir {RESULT} --tag {tag}"
    )


def main() -> None:
    commands = []
    for seed in (2024, 2025, 2026, 2027, 2028):
        for model in ("official_base", "mrmsgf"):
            commands.append(command(model, "original", 0, seed))
            commands.append(command(model, "point", 25, seed))
            commands.append(command(model, "point", 50, seed))
    OUT.write_text("\n".join(commands) + "\n")
    print(f"wrote {len(commands)} commands to {OUT}")


if __name__ == "__main__":
    main()
