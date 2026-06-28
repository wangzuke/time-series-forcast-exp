#!/usr/bin/env bash
# 用 GNU parallel / 自实现的 round-robin GPU 调度，运行 build_cmds.py 生成的命令清单。
# 用法：
#   bash scripts/run_experiments.sh scripts/cmds.txt 0,1,2,3,4,5,6,7 2
# 第二个参数是 GPU id 列表（逗号分隔），第三个参数是每张卡并发任务数。

set -uo pipefail

CMD_FILE=${1:-scripts/cmds.txt}
GPU_LIST=${2:-0}
PER_GPU=${3:-1}
LOG_DIR=${4:-logs}

mkdir -p "$LOG_DIR"

IFS=',' read -ra GPUS <<< "$GPU_LIST"
N_GPUS=${#GPUS[@]}
TOTAL_SLOTS=$((N_GPUS * PER_GPU))

echo "GPUs=${GPU_LIST}, per-gpu=${PER_GPU}, total slots=${TOTAL_SLOTS}"

# 读命令清单
mapfile -t CMDS < "$CMD_FILE"
N=${#CMDS[@]}
echo "total commands=${N}"

# 简易 round-robin：用后台进程 + wait -n 控制并发
run_one() {
    local gid=$1
    local idx=$2
    local cmd=$3
    local logfile="$LOG_DIR/run_$idx.log"
    CUDA_VISIBLE_DEVICES=$gid bash -c "$cmd" >> "$logfile" 2>&1
}

active=0
for i in "${!CMDS[@]}"; do
    gpu=${GPUS[$(( (i / PER_GPU) % N_GPUS ))]}
    cmd=${CMDS[$i]}
    if [[ -z "$cmd" ]]; then continue; fi
    echo "[$i / $N] GPU=$gpu :: ${cmd:0:160} ..."
    run_one $gpu $i "$cmd" &
    active=$((active+1))
    if (( active >= TOTAL_SLOTS )); then
        wait -n || true
        active=$((active-1))
    fi
done

wait
echo "done"
