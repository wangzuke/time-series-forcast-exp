#!/usr/bin/env bash
# Run post-hoc horizon decomposition after Stage C has a validated report.
set -euo pipefail

ROOT="/media/data1/wangzk/time-series-forcast-exp"
C_PID="${1:?usage: r0910_continue_stage_d.sh <stage_c_continuation_pid>}"
C_LOG="$ROOT/missing_ts_exp/logs/r0910_evidence_closure/stage_c_continuation.log"
LOG="$ROOT/missing_ts_exp/logs/r0910_evidence_closure/stage_d_continuation.log"

cd "$ROOT"
mkdir -p "$(dirname "$LOG")"
exec >> "$LOG" 2>&1
echo "[$(date --iso-8601=seconds)] waiting for Stage C PID $C_PID"
while kill -0 "$C_PID" 2>/dev/null; do
  sleep 60
done

if ! rg -q 'Stage C continuation completed' "$C_LOG"; then
  echo "[$(date --iso-8601=seconds)] Stage C failed; refusing Stage D"
  exit 1
fi

echo "[$(date --iso-8601=seconds)] starting Stage D horizon decomposition"
bash missing_ts_exp/scripts/r0910_run_stage_d_horizon.sh
/media/data1/wangzk/miniconda3/envs/hd-tts/bin/python missing_ts_exp/scripts/r0910_verify_stage_d_horizon.py
/media/data1/wangzk/miniconda3/envs/hd-tts/bin/python missing_ts_exp/scripts/r0910_write_stage_d_report.py
echo "[$(date --iso-8601=seconds)] Stage D continuation completed"
