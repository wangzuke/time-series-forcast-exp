#!/usr/bin/env bash
# Start Point comparisons only after the Block-T main-table expansion succeeds.
set -euo pipefail

ROOT="/media/data1/wangzk/time-series-forcast-exp"
A1_PID="${1:?usage: r0910_continue_stage_a2.sh <stage_a1_continuation_pid>}"
A1_LOG="$ROOT/missing_ts_exp/logs/r0910_evidence_closure/stage_a1_continuation.log"
LOG="$ROOT/missing_ts_exp/logs/r0910_evidence_closure/stage_a2_continuation.log"

cd "$ROOT"
mkdir -p "$(dirname "$LOG")"
exec >> "$LOG" 2>&1
echo "[$(date --iso-8601=seconds)] waiting for Stage A1 PID $A1_PID"
while kill -0 "$A1_PID" 2>/dev/null; do
  sleep 60
done

if ! rg -q 'Stage A1 continuation completed' "$A1_LOG"; then
  echo "[$(date --iso-8601=seconds)] Stage A1 failed; refusing Stage A2"
  exit 1
fi

echo "[$(date --iso-8601=seconds)] starting Stage A2"
bash missing_ts_exp/scripts/r0910_run_stage_a2.sh
echo "[$(date --iso-8601=seconds)] validating Stage A2"
/media/data1/wangzk/miniconda3/envs/hd-tts/bin/python missing_ts_exp/scripts/r0910_verify_stage_a2.py
echo "[$(date --iso-8601=seconds)] rendering Stage A2 report"
/media/data1/wangzk/miniconda3/envs/hd-tts/bin/python missing_ts_exp/scripts/r0910_write_stage_a2_report.py
echo "[$(date --iso-8601=seconds)] Stage A2 continuation completed"
