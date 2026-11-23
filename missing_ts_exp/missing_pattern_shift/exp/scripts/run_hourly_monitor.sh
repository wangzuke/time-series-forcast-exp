#!/usr/bin/env bash
set -euo pipefail
root_dir="$(cd "$(dirname "$0")/.." && pwd)"
python_bin="/media/data1/wangzk/miniconda3/envs/itransformer/bin/python"
manifest_path="${PRIMARY_MANIFEST:-$root_dir/configs/primary_jobs_2seeds.json}"
while :; do
  "$python_bin" "$root_dir/scripts/monitor_primary.py" --manifest "$manifest_path" || true
  "$python_bin" "$root_dir/scripts/validate_primary_results.py" --manifest "$manifest_path" || true
  "$python_bin" "$root_dir/scripts/aggregate_primary.py" --manifest "$manifest_path" || true
  "$python_bin" "$root_dir/scripts/plot_primary.py" || true
  if ! pgrep -f 'run_primary_worker.py' >/dev/null; then
    break
  fi
  sleep 3600
done
