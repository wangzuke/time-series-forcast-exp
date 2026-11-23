#!/usr/bin/env bash
set -u

EXP_ROOT="/media/data1/wangzk/time-series-forcast-exp/missing_ts_exp/missing_pattern_shift/exp"
PY_BIN="/media/data1/wangzk/miniconda3/envs/itransformer/bin/python"
MANIFEST="$EXP_ROOT/configs/s4m_traffic_electricity_stability_1115.json"
LOG="$EXP_ROOT/logs/s4m_traffic_electricity_stability_1115/queue_gpu7.log"
WORKER="$EXP_ROOT/scripts/run_traffic_stability_worker.py"
mkdir -p "$(dirname "$LOG")"

run_shard() {
    local baseline="$1"
    local phase="$2"
    local tag="queue_gpu7_${baseline}_${phase}"
    printf '\n[%s] START %s / %s\n' "$(date -Is)" "$baseline" "$phase" >> "$LOG"
    if "$PY_BIN" -u "$WORKER" --manifest "$MANIFEST" --gpu 7 --baseline "$baseline" \
        --phase "$phase" --status-tag "$tag" >> "$LOG" 2>&1; then
        printf '[%s] DONE %s / %s\n' "$(date -Is)" "$baseline" "$phase" >> "$LOG"
    else
        printf '[%s] FAILED %s / %s\n' "$(date -Is)" "$baseline" "$phase" >> "$LOG"
    fi
}

run_shard S4M traffic
run_shard S4M electricity_reference
run_shard S4M electricity_ablation
run_shard ChannelTokenFormer traffic
run_shard iTransformer traffic
run_shard PatchTST traffic
