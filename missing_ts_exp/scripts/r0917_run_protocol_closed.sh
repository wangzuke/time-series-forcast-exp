#!/usr/bin/env bash
set -euo pipefail

ROOT=/media/data1/wangzk/time-series-forcast-exp
PYTHON=/media/data1/wangzk/miniconda3/envs/hd-tts/bin/python
GPUS=${1:-0}
PER_GPU=${2:-1}
LOG_DIR="$ROOT/missing_ts_exp/logs/r0917_protocol_closed/workers"

cd "$ROOT"
"$PYTHON" missing_ts_exp/scripts/r0917_protocol_closed_p0.py
"$PYTHON" missing_ts_exp/scripts/r0917_build_protocol_closed_cmds.py
mkdir -p "$LOG_DIR"
bash missing_ts_exp/scripts/run_experiments.sh \
  missing_ts_exp/scripts/r0917_protocol_closed_cmds.txt "$GPUS" "$PER_GPU" "$LOG_DIR"
"$PYTHON" missing_ts_exp/scripts/r0917_verify_protocol_closed.py
"$PYTHON" missing_ts_exp/scripts/r0917_write_protocol_closed_report.py
