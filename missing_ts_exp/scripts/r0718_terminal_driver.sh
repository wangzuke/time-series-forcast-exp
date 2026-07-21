#!/usr/bin/env bash
set -euo pipefail

# User-terminal driver for the 0718 experiments.
#
# This is the no-Codex-authorization path: run it yourself in a normal terminal
# that can see the A800 GPUs.  It starts a host-side watcher, then runs the
# selected experiment phase using the existing scripts.
#
# Usage:
#   bash missing_ts_exp/scripts/r0718_terminal_driver.sh smoke
#   bash missing_ts_exp/scripts/r0718_terminal_driver.sh batch_probe
#   bash missing_ts_exp/scripts/r0718_terminal_driver.sh full
#   bash missing_ts_exp/scripts/r0718_terminal_driver.sh monitor_only

ROOT=/data/wangzuke/time-series-forecast-exp
RESULTS="$ROOT/missing_ts_exp/results/0718_block_hmbg"
STATUS_DIR="$RESULTS/status"
MODE="${1:-smoke}"
WATCH_INTERVAL="${R0718_WATCH_INTERVAL:-60}"

mkdir -p "$STATUS_DIR"

bash "$ROOT/missing_ts_exp/scripts/r0718_host_watch.sh" "$WATCH_INTERVAL" &
WATCH_PID=$!
echo "$WATCH_PID" > "$STATUS_DIR/driver_watch.pid"

cleanup() {
  touch "$STATUS_DIR/STOP_WATCHER"
  wait "$WATCH_PID" 2>/dev/null || true
  python "$ROOT/missing_ts_exp/scripts/r0718_collect_results.py" || true
}
trap cleanup EXIT

case "$MODE" in
  smoke)
    bash "$ROOT/missing_ts_exp/scripts/r0718_run_smoke.sh"
    ;;
  batch_probe)
    bash "$ROOT/missing_ts_exp/scripts/r0718_run_baselines.sh" batch_probe
    ;;
  full)
    bash "$ROOT/missing_ts_exp/scripts/r0718_run_baselines.sh" full
    ;;
  monitor_only)
    echo "[r0718 terminal driver] monitor_only running. Press Ctrl-C to stop."
    while true; do
      sleep 3600
    done
    ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Expected one of: smoke, batch_probe, full, monitor_only"
    exit 64
    ;;
esac
