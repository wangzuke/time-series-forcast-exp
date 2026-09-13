#!/usr/bin/env bash
set -euo pipefail

ROOT=/media/data1/wangzk/time-series-forcast-exp
cd "$ROOT"
PYTHON=/media/data1/wangzk/miniconda3/envs/hd-tts/bin/python

"$PYTHON" missing_ts_exp/scripts/r0908_build_formal_cmds.py
mkdir -p missing_ts_exp/logs/r0908_official_baselines/formal_workers
bash missing_ts_exp/scripts/run_experiments.sh \
  missing_ts_exp/scripts/r0908_formal_cmds.txt \
  0,1,2,3,4,5,6,7 1 \
  missing_ts_exp/logs/r0908_official_baselines/formal_workers
"$PYTHON" missing_ts_exp/scripts/r0908_verify_formal.py
