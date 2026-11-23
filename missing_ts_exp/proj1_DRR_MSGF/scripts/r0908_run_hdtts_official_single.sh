#!/usr/bin/env bash
set -euo pipefail

ROOT=/media/data1/wangzk/time-series-forcast-exp
HDTTS_ROOT="$ROOT/external_repro/official/hdtts"
PYTHON=/media/data1/wangzk/miniconda3/envs/hd-tts/bin/python

MODEL=${1:?usage: r0908_run_hdtts_official_single.sh <model> <seed> <out_dir>}
SEED=${2:?usage: r0908_run_hdtts_official_single.sh <model> <seed> <out_dir>}
OUT_DIR=${3:?usage: r0908_run_hdtts_official_single.sh <model> <seed> <out_dir>}

: "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES must be assigned by the worker pool}"
cd "$HDTTS_ROOT"
PYTHONPATH="$HDTTS_ROOT${PYTHONPATH:+:$PYTHONPATH}" MPLCONFIGDIR=/tmp/mpl-r0908 \
  "$PYTHON" experiments/run_realworld.py \
  "model=$MODEL" dataset=la dataset/mode=block_t \
  epochs=200 train_batches=300 patience=30 batch_size=32 workers=0 \
  "+run.seed=$SEED" "hydra.run.dir=$OUT_DIR"
