#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/wangzuke/time-series-forecast-exp
RESULTS="$ROOT/missing_ts_exp/results/0723_official_pguts_coarse_imputation"
RAW_LOG_DIR="$RESULTS/raw_logs"
OUT_LOG="${R0723_MONITOR_LOG:-$RESULTS/monitor_30min.log}"
INTERVAL="${R0723_MONITOR_INTERVAL:-1800}"

mkdir -p "$RESULTS"

snapshot() {
  local ts
  ts="$(date '+%Y-%m-%d %H:%M:%S %z')"
  {
    echo "================================================================================"
    echo "[r0723 monitor] $ts"
    echo
    echo "[controllers]"
    for f in "$RESULTS/reproduce_controller.log" "$RESULTS/coarse_controller.log"; do
      if [[ -f "$f" ]]; then
        echo "--- $(basename "$f")"
        tail -n 12 "$f" || true
      else
        echo "missing $(basename "$f")"
      fi
    done
    echo
    echo "[raw log counts]"
    printf "reproduce logs: "
    find "$RAW_LOG_DIR" -maxdepth 1 -type f -name 'reproduce_*.log' 2>/dev/null | wc -l
    printf "coarse logs: "
    find "$RAW_LOG_DIR" -maxdepth 1 -type f -name 'coarse_*.log' 2>/dev/null | wc -l
    printf "completed logs with Test MAE: "
    grep -Rsl "Test MAE:" "$RAW_LOG_DIR"/*.log 2>/dev/null | wc -l
    echo
    echo "[recent Test MAE]"
    grep -Rsh "Test MAE:" "$RAW_LOG_DIR"/*.log 2>/dev/null | tail -n 12 || true
    echo
    echo "[runner processes]"
    ps -eo pid,ppid,stat,etime,cmd | grep -E 'r0723_run_official|run_imputation|conda run' | grep -v grep || true
    echo
    echo "[gpu]"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>&1 || true
    echo
  } >> "$OUT_LOG"
}

finished() {
  grep -q "mode=reproduce finished" "$RESULTS/reproduce_controller.log" 2>/dev/null &&
    grep -q "mode=coarse finished" "$RESULTS/coarse_controller.log" 2>/dev/null
}

echo "[r0723 monitor] started at $(date '+%Y-%m-%d %H:%M:%S %z'), interval=${INTERVAL}s" >> "$OUT_LOG"

while true; do
  snapshot
  if finished; then
    echo "[r0723 monitor] experiments finished at $(date '+%Y-%m-%d %H:%M:%S %z')" >> "$OUT_LOG"
    exit 0
  fi
  sleep "$INTERVAL"
done
