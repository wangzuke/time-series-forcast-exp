#!/usr/bin/env bash
set -euo pipefail

# Host-side watcher for the 0718 experiments.
#
# Run this from a normal terminal that can see nvidia-smi.  It writes status
# files into the workspace so Codex can monitor progress from its restricted
# sandbox without requiring non-sandbox command execution.
#
# Usage:
#   bash missing_ts_exp/scripts/r0718_host_watch.sh 60
#
# Stop:
#   touch missing_ts_exp/results/0718_block_hmbg/status/STOP_WATCHER

ROOT=/data/wangzuke/time-series-forecast-exp
RESULTS="$ROOT/missing_ts_exp/results/0718_block_hmbg"
STATUS_DIR="$RESULTS/status"
LOG_DIR="$RESULTS/raw_logs"
INTERVAL="${1:-60}"
STOP_FILE="$STATUS_DIR/STOP_WATCHER"

mkdir -p "$STATUS_DIR" "$LOG_DIR" "$RESULTS/csv"
rm -f "$STOP_FILE"

echo "watcher_pid=$$" > "$STATUS_DIR/watcher.pid"
echo "started_at=$(date '+%Y-%m-%d %H:%M:%S %Z')" > "$STATUS_DIR/watcher_latest.txt"

while [[ ! -f "$STOP_FILE" ]]; do
  now="$(date '+%Y-%m-%d %H:%M:%S %Z')"

  {
    echo "time=$now"
    echo
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv || true
  } > "$STATUS_DIR/gpu_status_latest.txt" 2>&1

  {
    echo "time=$now"
    echo
    nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv || true
  } > "$STATUS_DIR/gpu_processes_latest.txt" 2>&1

  {
    echo "time=$now"
    echo
    ps -eo pid,ppid,stat,etime,pcpu,pmem,args \
      | grep -E 'run_realworld|run_realworld_patched|BiaTCGNet|main.py|test_forecasting.py|block_hmbg|0718_block_hmbg' \
      | grep -v grep || true
  } > "$STATUS_DIR/process_status_latest.txt" 2>&1

  {
    echo "time=$now"
    echo
    find "$LOG_DIR" -maxdepth 1 -type f -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' \
      | sort -r \
      | head -n 30 || true
  } > "$STATUS_DIR/log_status_latest.txt" 2>&1

  python "$ROOT/missing_ts_exp/scripts/r0718_collect_results.py" \
    > "$STATUS_DIR/collector_latest.txt" 2>&1 || true

  echo "last_heartbeat=$now" > "$STATUS_DIR/watcher_latest.txt"
  sleep "$INTERVAL"
done

echo "stopped_at=$(date '+%Y-%m-%d %H:%M:%S %Z')" >> "$STATUS_DIR/watcher_latest.txt"
