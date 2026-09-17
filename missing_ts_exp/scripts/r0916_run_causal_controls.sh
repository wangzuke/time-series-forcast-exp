#!/usr/bin/env bash
set -euo pipefail

ROOT=/media/data1/wangzk/time-series-forcast-exp
PYTHON=/media/data1/wangzk/miniconda3/envs/hd-tts/bin/python
GPUS=${1:-0}
PER_GPU=${2:-1}
LOG_DIR="$ROOT/missing_ts_exp/logs/r0916_causal_controls/workers"

cd "$ROOT"
"$PYTHON" missing_ts_exp/scripts/r0916_build_causal_control_cmds.py
mkdir -p "$LOG_DIR"
bash missing_ts_exp/scripts/run_experiments.sh \
  missing_ts_exp/scripts/r0916_causal_control_cmds.txt "$GPUS" "$PER_GPU" "$LOG_DIR"
"$PYTHON" missing_ts_exp/scripts/r0916_verify_causal_controls.py
"$PYTHON" missing_ts_exp/scripts/r0916_write_causal_control_report.py
