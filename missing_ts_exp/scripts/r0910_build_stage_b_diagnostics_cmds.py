#!/usr/bin/env python3
"""Build full-test reliability diagnostics for the validated full-model runs."""
from pathlib import Path


ROOT = Path("/media/data1/wangzk/time-series-forcast-exp")
OUT = ROOT / "missing_ts_exp/scripts/r0910_stage_b_diagnostics_cmds.txt"
OLD = ROOT / "missing_ts_exp/results/r0908_official_baselines/controlled"
NEW = ROOT / "missing_ts_exp/results/r0910_evidence_closure/stage_b_ablation"
RESULT = ROOT / "missing_ts_exp/results/r0910_evidence_closure/stage_b_diagnostics"
PYTHON = "/media/data1/wangzk/miniconda3/envs/hd-tts/bin/python"


def result_json(rate: int, seed: int) -> Path:
    stem = "mrmsgf" if seed <= 2026 else "full"
    root = OLD if seed <= 2026 else NEW
    return root / f"{stem}_METR-LA_block_t_r{rate}_seed{seed}.json"


def main() -> None:
    commands = []
    for rate in (50, 70):
        for seed in (2024, 2025, 2026, 2027, 2028):
            output = RESULT / f"full_METR-LA_block_t_r{rate}_seed{seed}_diagnostics.json"
            commands.append(
                f"{PYTHON} -m missing_ts_exp.scripts.r0910_collect_mrmsgf_diagnostics "
                f"--result_json {result_json(rate, seed)} --device cuda --out_path {output}"
            )
    OUT.write_text("\n".join(commands) + "\n")
    print(f"wrote {len(commands)} commands to {OUT}")


if __name__ == "__main__":
    main()
