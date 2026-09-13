#!/usr/bin/env bash
# Start A1 only after the Stage-B continuation generated its verified report.
set -euo pipefail

ROOT="/media/data1/wangzk/time-series-forcast-exp"
STAGE_B_PID="${1:?usage: r0910_continue_stage_a1.sh <stage_b_continuation_pid>}"
STAGE_B_LOG="$ROOT/missing_ts_exp/logs/r0910_evidence_closure/stage_b_continuation.log"
LOG="$ROOT/missing_ts_exp/logs/r0910_evidence_closure/stage_a1_continuation.log"

cd "$ROOT"
mkdir -p "$(dirname "$LOG")"
exec >> "$LOG" 2>&1
echo "[$(date --iso-8601=seconds)] waiting for Stage B continuation PID $STAGE_B_PID"
while kill -0 "$STAGE_B_PID" 2>/dev/null; do
  sleep 60
done

if ! rg -q 'Stage B continuation completed' "$STAGE_B_LOG"; then
  echo "[$(date --iso-8601=seconds)] Stage B continuation failed; refusing Stage A1"
  exit 1
fi

echo "[$(date --iso-8601=seconds)] starting Stage A1"
bash missing_ts_exp/scripts/r0910_run_stage_a1.sh
echo "[$(date --iso-8601=seconds)] validating Stage A1"
/media/data1/wangzk/miniconda3/envs/hd-tts/bin/python missing_ts_exp/scripts/r0910_verify_stage_a1.py
echo "[$(date --iso-8601=seconds)] rendering Stage A1 report"
/media/data1/wangzk/miniconda3/envs/hd-tts/bin/python missing_ts_exp/scripts/r0910_write_stage_a1_report.py
echo "[$(date --iso-8601=seconds)] Stage A1 continuation completed"
