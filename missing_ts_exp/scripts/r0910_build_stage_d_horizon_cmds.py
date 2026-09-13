#!/usr/bin/env python3
"""Build post-hoc five-seed horizon evaluation commands for Stage D."""
from pathlib import Path


ROOT = Path("/media/data1/wangzk/time-series-forcast-exp")
OUT = ROOT / "missing_ts_exp/scripts/r0910_stage_d_horizon_cmds.txt"
OLD = ROOT / "missing_ts_exp/results/r0908_official_baselines/controlled"
NEW = ROOT / "missing_ts_exp/results/r0910_evidence_closure/stage_b_ablation"
RESULT = ROOT / "missing_ts_exp/results/r0910_evidence_closure/stage_d_horizon"
PYTHON = "/media/data1/wangzk/miniconda3/envs/hd-tts/bin/python"


def source(model: str, rate: int, seed: int) -> Path:
    root = OLD if seed <= 2026 else NEW
    stem = model if seed <= 2026 or model == "official_base" else "full"
    return root / f"{stem}_METR-LA_block_t_r{rate}_seed{seed}.json"


def main() -> None:
    commands = []
    for rate in (50, 70):
        for seed in (2024, 2025, 2026, 2027, 2028):
            for model in ("official_base", "mrmsgf"):
                out = RESULT / f"{model}_METR-LA_block_t_r{rate}_seed{seed}_horizon.json"
                commands.append(
                    f"{PYTHON} -m missing_ts_exp.scripts.r0910_collect_horizon_metrics "
                    f"--result_json {source(model, rate, seed)} --device cuda --out_path {out}"
                )
    OUT.write_text("\n".join(commands) + "\n")
    print(f"wrote {len(commands)} commands to {OUT}")


if __name__ == "__main__":
    main()
