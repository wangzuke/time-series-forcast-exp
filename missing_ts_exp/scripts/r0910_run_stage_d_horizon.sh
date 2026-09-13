#!/usr/bin/env bash
set -euo pipefail

ROOT="/media/data1/wangzk/time-series-forcast-exp"
cd "$ROOT"
mkdir -p missing_ts_exp/logs/r0910_evidence_closure/stage_d_horizon
/media/data1/wangzk/miniconda3/envs/hd-tts/bin/python missing_ts_exp/scripts/r0910_build_stage_d_horizon_cmds.py
bash missing_ts_exp/scripts/run_experiments.sh \
  missing_ts_exp/scripts/r0910_stage_d_horizon_cmds.txt \
  0,1,2,3,4,5,6,7 \
  1 \
  missing_ts_exp/logs/r0910_evidence_closure/stage_d_horizon
