#!/usr/bin/env bash
# Resume the evidence pipeline after Stage-B diagnostics failed before writing.
set -euo pipefail

ROOT="/media/data1/wangzk/time-series-forcast-exp"
PYTHON="/media/data1/wangzk/miniconda3/envs/hd-tts/bin/python"
LOG="$ROOT/missing_ts_exp/logs/r0910_evidence_closure/r0911_resume_after_stage_b.log"

cd "$ROOT"
mkdir -p "$(dirname "$LOG")"
exec >> "$LOG" 2>&1

echo "[$(date --iso-8601=seconds)] rerunning fixed Stage-B diagnostics"
bash missing_ts_exp/scripts/r0910_run_stage_b_diagnostics.sh
"$PYTHON" missing_ts_exp/scripts/r0910_write_stage_b_report.py
echo "[$(date --iso-8601=seconds)] Stage B resumed report completed"

echo "[$(date --iso-8601=seconds)] starting Stage A1"
bash missing_ts_exp/scripts/r0910_run_stage_a1.sh
"$PYTHON" missing_ts_exp/scripts/r0910_verify_stage_a1.py
"$PYTHON" missing_ts_exp/scripts/r0910_write_stage_a1_report.py
echo "[$(date --iso-8601=seconds)] Stage A1 completed"

echo "[$(date --iso-8601=seconds)] starting Stage A2"
bash missing_ts_exp/scripts/r0910_run_stage_a2.sh
"$PYTHON" missing_ts_exp/scripts/r0910_verify_stage_a2.py
"$PYTHON" missing_ts_exp/scripts/r0910_write_stage_a2_report.py
echo "[$(date --iso-8601=seconds)] Stage A2 completed"

echo "[$(date --iso-8601=seconds)] generating formal AQI36 resources"
"$PYTHON" -m missing_ts_exp.scripts.r0905_prepare_blockt_resources \
  --seeds 2024,2025,2026,2027,2028 --rates 0.30,0.50,0.70 --aqi_point_rates 0.25,0.50
test -f missing_ts_exp/resources/r0905_blockt/masks/AQI36/point_r50_seed2028_observed.npy

echo "[$(date --iso-8601=seconds)] starting Stage C"
bash missing_ts_exp/scripts/r0910_run_stage_c.sh
"$PYTHON" missing_ts_exp/scripts/r0910_verify_stage_c.py
"$PYTHON" missing_ts_exp/scripts/r0910_write_stage_c_report.py
echo "[$(date --iso-8601=seconds)] Stage C completed"

echo "[$(date --iso-8601=seconds)] starting Stage D"
bash missing_ts_exp/scripts/r0910_run_stage_d_horizon.sh
"$PYTHON" missing_ts_exp/scripts/r0910_verify_stage_d_horizon.py
"$PYTHON" missing_ts_exp/scripts/r0910_write_stage_d_report.py
echo "[$(date --iso-8601=seconds)] evidence pipeline completed"
