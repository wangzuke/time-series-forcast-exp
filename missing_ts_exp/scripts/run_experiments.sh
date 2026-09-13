#!/usr/bin/env bash
# 自实现的 GPU worker-pool 调度，运行 build_cmds.py 生成的命令清单。
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

# A finished worker immediately receives the next command on the same GPU.  This
# avoids stranding a GPU when another, slower GPU owns the next round-robin slot.
run_one() {
    local gid=$1
    local idx=$2
    local cmd=$3
    local logfile="$LOG_DIR/run_$idx.log"
    CUDA_VISIBLE_DEVICES=$gid bash -c "$cmd" >> "$logfile" 2>&1
}

declare -A PID_GPU
next=0
active=0
failed=0

launch_next() {
    local gpu=$1
    local idx cmd pid
    while (( next < N )); do
        idx=$next
        next=$((next + 1))
        cmd=${CMDS[$idx]}
        [[ -z "$cmd" ]] && continue
        echo "[$idx / $N] GPU=$gpu :: ${cmd:0:160} ..."
        run_one "$gpu" "$idx" "$cmd" &
        pid=$!
        PID_GPU[$pid]=$gpu
        active=$((active + 1))
        return 0
    done
    return 1
}

# Fill the initial worker pool. Multiple slots per GPU are still supported.
for ((slot = 0; slot < TOTAL_SLOTS && next < N; slot++)); do
    launch_next "${GPUS[$((slot % N_GPUS))]}" || break
done

while (( active > 0 )); do
    finished_pid=""
    if wait -n -p finished_pid; then
        :
    else
        failed=1
        echo "worker ${finished_pid:-unknown} failed" >&2
    fi
    gpu=${PID_GPU[$finished_pid]:-}
    unset 'PID_GPU[$finished_pid]'
    active=$((active - 1))
    if [[ -z "$gpu" ]]; then
        echo "unable to identify the GPU for completed worker ${finished_pid:-unknown}" >&2
        failed=1
        continue
    fi
    launch_next "$gpu" || true
done

if (( failed )); then
    echo "one or more experiment commands failed" >&2
    exit 1
fi
echo "done"
