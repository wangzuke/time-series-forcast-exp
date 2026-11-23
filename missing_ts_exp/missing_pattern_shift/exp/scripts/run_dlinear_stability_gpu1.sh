#!/usr/bin/env bash
set -u

EXP_ROOT="/media/data1/wangzk/time-series-forcast-exp/missing_ts_exp/missing_pattern_shift/exp"
PY_BIN="/media/data1/wangzk/miniconda3/envs/itransformer/bin/python"
MANIFEST="$EXP_ROOT/configs/s4m_traffic_electricity_stability_1115.json"
LOG="$EXP_ROOT/logs/s4m_traffic_electricity_stability_1115/queue_gpu1_dlinear.log"
WORKER="$EXP_ROOT/scripts/run_traffic_stability_worker.py"
mkdir -p "$(dirname "$LOG")"
printf '[%s] START DLinear / Traffic\n' "$(date -Is)" >> "$LOG"
if "$PY_BIN" -u "$WORKER" --manifest "$MANIFEST" --gpu 1 --baseline DLinear \
    --phase traffic --status-tag queue_gpu1_dlinear_traffic >> "$LOG" 2>&1; then
    printf '[%s] DONE DLinear / Traffic\n' "$(date -Is)" >> "$LOG"
else
    printf '[%s] FAILED DLinear / Traffic\n' "$(date -Is)" >> "$LOG"
fi
