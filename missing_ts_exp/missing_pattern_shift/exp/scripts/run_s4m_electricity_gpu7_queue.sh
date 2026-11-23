#!/usr/bin/env bash
set -u

EXP_ROOT="/media/data1/wangzk/time-series-forcast-exp/missing_ts_exp/missing_pattern_shift/exp"
PY_BIN="/media/data1/wangzk/miniconda3/envs/itransformer/bin/python"
MANIFEST="$EXP_ROOT/configs/s4m_traffic_electricity_stability_1115.json"
LOG="$EXP_ROOT/logs/s4m_traffic_electricity_stability_1115/queue_gpu7_electricity_only.log"
WORKER="$EXP_ROOT/scripts/run_traffic_stability_worker.py"
mkdir -p "$(dirname "$LOG")"

run_shard() {
    local phase="$1"
    local tag="queue_gpu7_S4M_${phase}_1117"
    printf '\n[%s] START S4M / %s\n' "$(date -Is)" "$phase" >> "$LOG"
    if "$PY_BIN" -u "$WORKER" --manifest "$MANIFEST" --gpu 7 --baseline S4M \
        --phase "$phase" --status-tag "$tag" >> "$LOG" 2>&1; then
        printf '[%s] DONE S4M / %s\n' "$(date -Is)" "$phase" >> "$LOG"
    else
        printf '[%s] FAILED S4M / %s\n' "$(date -Is)" "$phase" >> "$LOG"
        exit 1
    fi
}

run_shard electricity_reference
run_shard electricity_ablation
