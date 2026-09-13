#!/usr/bin/env bash
# Generate the final fixed AQI36 resources and run C after A2 is validated.
set -euo pipefail

ROOT="/media/data1/wangzk/time-series-forcast-exp"
A2_PID="${1:?usage: r0910_continue_stage_c.sh <stage_a2_continuation_pid>}"
A2_LOG="$ROOT/missing_ts_exp/logs/r0910_evidence_closure/stage_a2_continuation.log"
LOG="$ROOT/missing_ts_exp/logs/r0910_evidence_closure/stage_c_continuation.log"
PYTHON="/media/data1/wangzk/miniconda3/envs/hd-tts/bin/python"

cd "$ROOT"
mkdir -p "$(dirname "$LOG")"
exec >> "$LOG" 2>&1
echo "[$(date --iso-8601=seconds)] waiting for Stage A2 PID $A2_PID"
while kill -0 "$A2_PID" 2>/dev/null; do
  sleep 60
done

if ! rg -q 'Stage A2 continuation completed' "$A2_LOG"; then
  echo "[$(date --iso-8601=seconds)] Stage A2 failed; refusing Stage C"
  exit 1
fi

echo "[$(date --iso-8601=seconds)] generating five-seed formal resources"
"$PYTHON" -m missing_ts_exp.scripts.r0905_prepare_blockt_resources \
  --seeds 2024,2025,2026,2027,2028 --rates 0.30,0.50,0.70 --aqi_point_rates 0.25,0.50
test -f missing_ts_exp/resources/r0905_blockt/masks/AQI36/point_r50_seed2028_observed.npy
echo "[$(date --iso-8601=seconds)] starting Stage C"
bash missing_ts_exp/scripts/r0910_run_stage_c.sh
echo "[$(date --iso-8601=seconds)] validating Stage C"
"$PYTHON" missing_ts_exp/scripts/r0910_verify_stage_c.py
echo "[$(date --iso-8601=seconds)] rendering Stage C report"
"$PYTHON" missing_ts_exp/scripts/r0910_write_stage_c_report.py
echo "[$(date --iso-8601=seconds)] Stage C continuation completed"
