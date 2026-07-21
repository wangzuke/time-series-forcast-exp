#!/usr/bin/env bash
set -euo pipefail

# Phase 4 runner: Block-HMBGNet main experiments (實驗計劃0718.md 7.1/7.2) plus
# priority ablations (7.3). Baselines (HD-TTS-AMP, BiTGraph) for all of these
# configs were already produced by Phase 2 (r0718_run_baselines.sh) and
# Phase 2b (r0718_run_baselines_2b.sh) -- this script only launches the fusion
# model, model=block_hmbg, through the same run_realworld_patched.py entry
# point used for the HD-TTS baseline.
#
# Model hparams (external_repro/hdtts/config/model/block_hmbg.yaml, wired
# into external_repro/hdtts/lib/nn/models/block_hmbg_model.py):
#   use_block_gate      -- block-aware readout attention gate
#   use_coarse_graph     -- coarse-scale mask-biased graph refinement
#   use_boundary_graph   -- boundary-distance-gated fine-scale graph refinement
#   boundary_gate_static -- when true, boundary graph applies with constant
#                           weight instead of the boundary-distance gate
#                           (the "graph everywhere" ablation)
# All four are overridden via hydra dotted-path CLI overrides
# (model.hparams.<name>=<value>), confirmed to work because run_realworld.py
# does `model_kwargs.update(cfg.model.hparams)`.
#
# Main-experiment variant (7.1/7.2) = full model:
#   use_block_gate=true use_coarse_graph=true use_boundary_graph=true
#   boundary_gate_static=false
#
# Ablation variants (7.3), run only on the plan's priority configs
# (METR-LA/PEMS-BAY block_t 0.7/0.8, METR-LA/PEMS-BAY block_st 0.8):
#   wo_gate           gate=false coarse=true  boundary=true  static=false
#   wo_coarse         gate=true  coarse=false boundary=true  static=false
#   wo_boundary       gate=true  coarse=true  boundary=false static=false
#   graph_everywhere  gate=true  coarse=true  boundary=true  static=true
#   fine_only         gate=false coarse=false boundary=true  static=false
#   coarse_only       gate=false coarse=true  boundary=false static=false
# (the "main"/full row itself is not re-run for the ablation table --
# ablation.csv's delta_mae_vs_full should reference the corresponding main
# Block-HMBGNet run produced by this same script.)
#
# Must be run AFTER Phase 2b has released GPUs 1-7 -- this script does its
# own independent GPU bookkeeping starting from "all of 1-7 free".
#
# Usage:
#   bash missing_ts_exp/scripts/r0718_run_phase4.sh
#
# Optional environment overrides:
#   R0718_HMBG_BATCH=128
#   R0718_EPOCHS=200
#   R0718_HDTTS_TRAIN_BATCHES=300
#   R0718_HDTTS_PATIENCE=30

ROOT=/data/wangzuke/time-series-forecast-exp
RESULTS="$ROOT/missing_ts_exp/results/0718_block_hmbg"
LOG_DIR="$RESULTS/raw_logs"
PID_DIR="$RESULTS/pids"
CKPT_DIR="$RESULTS/checkpoints"

mkdir -p "$LOG_DIR" "$PID_DIR" "$CKPT_DIR"

if ! nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader > "$LOG_DIR/gpu_preflight_phase4.log" 2>&1; then
  echo "[r0718 phase4] GPU preflight failed; refusing to start GPU experiments."
  cat "$LOG_DIR/gpu_preflight_phase4.log"
  exit 2
fi

GPUS=(1 2 3 4 5 6 7)
free_gpus=("${GPUS[@]}")
declare -A pid_gpu
failed_jobs=0

acquire_gpu_wait() {
  while (( ${#free_gpus[@]} == 0 )); do
    local done_pid
    if wait -n -p done_pid; then
      :
    else
      failed_jobs=$((failed_jobs + 1))
    fi
    free_gpus+=("${pid_gpu[$done_pid]}")
    unset 'pid_gpu[$done_pid]'
  done
}

launch_job() {
  local workdir="$1"
  local name="$2"
  shift 2

  acquire_gpu_wait
  local gpu="${free_gpus[0]}"
  free_gpus=("${free_gpus[@]:1}")
  local log="$LOG_DIR/${name}.log"

  echo "[r0718 phase4] launch gpu=${gpu} name=${name} cwd=${workdir}"
  (
    cd "$workdir"
    CUDA_VISIBLE_DEVICES="$gpu" "$@"
  ) > "$log" 2>&1 &

  local pid=$!
  echo "$pid" > "$PID_DIR/${name}.pid"
  pid_gpu[$pid]="$gpu"
}

run_block_hmbg() {
  local dataset="$1"
  local mode="$2"
  local variant_tag="$3"
  local gate="$4"
  local coarse="$5"
  local boundary="$6"
  local static="$7"
  local batch="$8"
  local epochs="$9"
  local train_batches="${10}"
  local patience="${11}"
  local name="block_hmbg_${variant_tag}_${dataset}_${mode}_h24_b${batch}_e${epochs}"
  local run_dir="$CKPT_DIR/$name"

  # boundary_gate_static is not declared in config/model/block_hmbg.yaml's
  # struct (only the constructor default, False, exists there implicitly),
  # so Hydra requires the `+` add-key prefix -- and only when it differs
  # from that default, since re-adding an already-present key with `+`
  # would itself error.
  local static_override=()
  if [[ "$static" == "true" ]]; then
    static_override=(+model.hparams.boundary_gate_static=true)
  fi

  launch_job "$ROOT/external_repro/hdtts" "$name" \
    conda run -n hd-tts python run_realworld_patched.py \
      model=block_hmbg \
      dataset="$dataset" \
      dataset/mode="$mode" \
      window=24 \
      horizon=24 \
      batch_size="$batch" \
      epochs="$epochs" \
      train_batches="$train_batches" \
      patience="$patience" \
      model.hparams.use_block_gate="$gate" \
      model.hparams.use_coarse_graph="$coarse" \
      model.hparams.use_boundary_graph="$boundary" \
      "${static_override[@]}" \
      hydra.run.dir="$run_dir"
}

HMBG_BATCH=${R0718_HMBG_BATCH:-128}
EPOCHS=${R0718_EPOCHS:-200}
TRAIN_BATCHES=${R0718_HDTTS_TRAIN_BATCHES:-300}
PATIENCE=${R0718_HDTTS_PATIENCE:-30}

# --- 7.1/7.2 main experiments: full model (main variant) ---
for rate in 30 50 60 70 80; do
  for ds in la bay; do
    run_block_hmbg "$ds" "block_t_${rate}" main true true true false \
      "$HMBG_BATCH" "$EPOCHS" "$TRAIN_BATCHES" "$PATIENCE"
    run_block_hmbg "$ds" "block_st_${rate}" main true true true false \
      "$HMBG_BATCH" "$EPOCHS" "$TRAIN_BATCHES" "$PATIENCE"
  done
done

for rate in 50 70 80; do
  for ds in la bay; do
    run_block_hmbg "$ds" "point_${rate}" main true true true false \
      "$HMBG_BATCH" "$EPOCHS" "$TRAIN_BATCHES" "$PATIENCE"
  done
done

# --- 7.3 priority ablations ---
# variant_tag gate coarse boundary static
ABLATIONS=(
  "wo_gate false true true false"
  "wo_coarse true false true false"
  "wo_boundary true true false false"
  "graph_everywhere true true true true"
  "fine_only false false true false"
  "coarse_only false true false false"
)

PRIORITY_CONFIGS=(
  "la block_t_70"
  "la block_t_80"
  "bay block_t_70"
  "bay block_t_80"
  "la block_st_80"
  "bay block_st_80"
)

for cfg in "${PRIORITY_CONFIGS[@]}"; do
  read -r ds mode <<< "$cfg"
  for ablation in "${ABLATIONS[@]}"; do
    read -r variant_tag gate coarse boundary static <<< "$ablation"
    run_block_hmbg "$ds" "$mode" "$variant_tag" "$gate" "$coarse" "$boundary" "$static" \
      "$HMBG_BATCH" "$EPOCHS" "$TRAIN_BATCHES" "$PATIENCE"
  done
done

while (( ${#pid_gpu[@]} > 0 )); do
  done_pid=""
  if wait -n -p done_pid; then
    :
  else
    failed_jobs=$((failed_jobs + 1))
  fi
  unset 'pid_gpu[$done_pid]'
done

echo "[r0718 phase4] all jobs finished; failed_jobs=${failed_jobs}"
python "$ROOT/missing_ts_exp/scripts/r0718_monitor.py" --tail-lines 30

if (( failed_jobs > 0 )); then
  exit 1
fi
