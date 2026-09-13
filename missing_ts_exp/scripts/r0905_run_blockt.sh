#!/usr/bin/env bash
set -euo pipefail

CMDS="${1:-missing_ts_exp/scripts/r0905_blockt_smoke_cmds.txt}"
GPUS="${2:-0}"
JOBS_PER_GPU="${3:-1}"
LOG_DIR="${4:-missing_ts_exp/logs/r0905_blockt}"

bash missing_ts_exp/scripts/run_experiments.sh "$CMDS" "$GPUS" "$JOBS_PER_GPU" "$LOG_DIR"
