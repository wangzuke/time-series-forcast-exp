#!/usr/bin/env bash
# Continue Stage B only after the currently running 58-job matrix succeeds.
set -euo pipefail

ROOT="/media/data1/wangzk/time-series-forcast-exp"
MASTER_PID="${1:?usage: r0910_continue_after_stage_b.sh <stage_b_master_pid>}"
MASTER_LOG="$ROOT/missing_ts_exp/logs/r0910_evidence_closure/stage_b_master.log"
CONTINUATION_LOG="$ROOT/missing_ts_exp/logs/r0910_evidence_closure/stage_b_continuation.log"

cd "$ROOT"
mkdir -p "$(dirname "$CONTINUATION_LOG")"
exec >> "$CONTINUATION_LOG" 2>&1
echo "[$(date --iso-8601=seconds)] waiting for Stage B PID $MASTER_PID"
while kill -0 "$MASTER_PID" 2>/dev/null; do
  sleep 60
done

if ! rg -qx 'done' "$MASTER_LOG"; then
  echo "[$(date --iso-8601=seconds)] Stage B did not finish successfully; refusing continuation"
  exit 1
fi

echo "[$(date --iso-8601=seconds)] Stage B main matrix succeeded; running overlap ablation"
bash missing_ts_exp/scripts/r0910_run_stage_b_overlap.sh
echo "[$(date --iso-8601=seconds)] validating Stage B"
/media/data1/wangzk/miniconda3/envs/hd-tts/bin/python missing_ts_exp/scripts/r0910_verify_stage_b.py
echo "[$(date --iso-8601=seconds)] collecting full-model diagnostics"
bash missing_ts_exp/scripts/r0910_run_stage_b_diagnostics.sh
echo "[$(date --iso-8601=seconds)] rendering Stage B report"
/media/data1/wangzk/miniconda3/envs/hd-tts/bin/python missing_ts_exp/scripts/r0910_write_stage_b_report.py
echo "[$(date --iso-8601=seconds)] Stage B continuation completed"
