#!/usr/bin/env bash
set -euo pipefail

ROOT="/media/data1/wangzk/time-series-forcast-exp"
COMMANDS="$ROOT/missing_ts_exp/scripts/r0916_critical_control_cmds.txt"
LOG_DIR="$ROOT/missing_ts_exp/logs/r0916_critical_controls"
MAX_WORKERS="${1:-8}"

mkdir -p "$LOG_DIR"
mapfile -t jobs < "$COMMANDS"
declare -a pids=()

launch() {
    local index="$1"
    local gpu=$((index % MAX_WORKERS))
    local log="$LOG_DIR/$(printf '%02d' "$index").log"
    (
        export CUDA_VISIBLE_DEVICES="$gpu"
        cd "$ROOT"
        echo "[$(date -Is)] GPU $gpu: ${jobs[index]}"
        ${jobs[index]}
    ) >"$log" 2>&1 &
    pids+=("$!")
}

for index in "${!jobs[@]}"; do
    while (( ${#pids[@]} >= MAX_WORKERS )); do
        wait -n
        alive=()
        for pid in "${pids[@]}"; do
            kill -0 "$pid" 2>/dev/null && alive+=("$pid")
        done
        pids=("${alive[@]}")
    done
    launch "$index"
done

for pid in "${pids[@]}"; do
    wait "$pid"
done
