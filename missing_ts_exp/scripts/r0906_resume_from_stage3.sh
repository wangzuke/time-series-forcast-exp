#!/usr/bin/env bash
set -euo pipefail

PYTHON=/media/data1/wangzk/miniconda3/envs/hd-tts/bin/python
LOG_ROOT=missing_ts_exp/logs/r0906_mrmsgf
RESULT_ROOT=missing_ts_exp/results/r0906_mrmsgf

validate_results() {
  local result_dir=$1
  local expected_count=$2
  "$PYTHON" -c '
import json, math, pathlib, sys
root = pathlib.Path(sys.argv[1])
expected = int(sys.argv[2])
files = sorted(root.glob("*.json"))
if len(files) != expected:
    raise SystemExit(f"expected {expected} result files under {root}, found {len(files)}")
for path in files:
    row = json.loads(path.read_text())
    test = row.get("test", {})
    for key in ("loss", "mae", "rmse", "mape"):
        value = test.get(key)
        if value is None or not math.isfinite(float(value)):
            raise SystemExit(f"{path}: invalid test.{key}={value}")
print(f"validated {len(files)} result files under {root}")
' "$result_dir" "$expected_count"
}

# Resume only after proving the interrupted pipeline's completed stages are valid.
validate_results "$RESULT_ROOT/stage1_core" 4
validate_results "$RESULT_ROOT/stage2_blockt_all" 18

bash missing_ts_exp/scripts/r0906_run_mrmsgf.sh \
  missing_ts_exp/scripts/r0906_mrmsgf_stage3_aux_cmds.txt \
  0,1,2,3,4,5,6,7 1 "$LOG_ROOT/stage3_aux"
validate_results "$RESULT_ROOT/stage3_aux" 30

bash missing_ts_exp/scripts/r0906_run_mrmsgf.sh \
  missing_ts_exp/scripts/r0906_mrmsgf_ablation_cmds.txt \
  0,1,2,3,4,5,6,7 1 "$LOG_ROOT/ablation"
validate_results "$RESULT_ROOT/ablation" 30

echo "R0906 resumed experiment pipeline completed successfully"
