#!/usr/bin/env bash
set -euo pipefail

ROOT=/media/data1/wangzk/time-series-forcast-exp
BITGRAPH_ROOT="$ROOT/external_repro/official/bitgraph"
PYTHON=/media/data1/wangzk/miniconda3/envs/hd-tts/bin/python
RATE=${1:?usage: r0908_run_bitgraph_official_replicates.sh <mask_ratio> <out_dir>}
OUT_DIR=${2:?usage: r0908_run_bitgraph_official_replicates.sh <mask_ratio> <out_dir>}
EPOCHS=${3:-200}
REPLICATES=${4:-3}

: "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES must be assigned by the worker pool}"
mkdir -p "$OUT_DIR"
cd "$BITGRAPH_ROOT"

for replicate in $(seq 1 "$REPLICATES"); do
  run_dir="$OUT_DIR/replicate${replicate}"
  mkdir -p "$run_dir"
  # The upstream code does not seed NumPy/Torch from --seed. The argument is
  # retained only as a run label; source code and its random masking logic stay unchanged.
  "$PYTHON" main.py --epochs "$EPOCHS" --dataset Metr --dataset-name Metr \
    --mask_ratio "$RATE" --seq_len 24 --pred_len 24 --horizon 24 \
    --seed "$replicate" >"$run_dir/train.log" 2>&1
  "$PYTHON" test_forecasting.py --epochs "$EPOCHS" --dataset Metr --dataset-name Metr \
    --mask_ratio "$RATE" --seq_len 24 --pred_len 24 --horizon 24 \
    --seed "$replicate" >"$run_dir/test.log" 2>&1
  cp "output_BiaTCGNet_Metr_miss${RATE}_prediction/best.pth" "$run_dir/best.pth"
  printf 'repo_commit=4f1d05bcc20bb3f5084bd1facc995478f84a40ed\nmask_ratio=%s\nreplicate=%s\n' \
    "$RATE" "$replicate" >"$run_dir/metadata.txt"
done
