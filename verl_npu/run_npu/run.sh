#!/bin/bash
# 顶层入口 — 与 llm_ft_longtime/run.sh 同构:
#   进程1(后台): check-point 增量上传;  进程2(阻塞): GRPO 训练。
# 用法: bash run.sh experiments/qwendrive/collision_risk_grpo_01/

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
CONDA_PATH=${CONDA_PATH:-"/home/ma-user/anaconda3/etc/profile.d/conda.sh"}
CONDA_ENV=${CONDA_ENV:-"verl_npu"}

if [ -f "$CONDA_PATH" ]; then
    source "$CONDA_PATH"
    conda activate "$CONDA_ENV" || { echo "Error: conda activate $CONDA_ENV 失败 (环境不存在?)"; exit 1; }
elif [ -n "${VC_TASK_INDEX:-}" ] || [ -n "${VC_WORKER_HOSTS:-}" ]; then
    # ModelArts 平台上 conda 必须可用, 否则会静默落到错误的 python 环境
    echo "Error: conda.sh not found at $CONDA_PATH"
    exit 1
else
    echo "Warning: conda.sh not found at $CONDA_PATH, 使用当前 python 环境 (本地调试)"
fi
which python

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"

# 定义清理函数，接收信号名称作为参数
cleanup() {
    local signal_name=$1
    echo -e "\n[Monitor] 捕捉到异常信号: ${signal_name}"
    if [ -n "$PY1_PID" ]; then
        echo "[Monitor] 正在杀死进程 1 (PID: $PY1_PID)..."
        kill $PY1_PID 2>/dev/null
    fi
    echo "[Monitor] 清理完成，脚本退出。"
    exit 1
}

trap 'cleanup SIGINT' SIGINT
trap 'cleanup SIGTERM' SIGTERM
trap 'cleanup SIGHUP' SIGHUP

# 1. 启动进程 1 (非阻塞模式 - 上传check-point)
echo "[Step 1] 启动 check-point上传脚本"
python upload_check-point.py --experiment_dir "$1" &
PY1_PID=$!

# 2. 启动进程 2 (阻塞模式 - GRPO 训练)
echo "[Step 2] 启动进程 2 (阻塞模式 训练)..."
python run.py --experiment_dir "$1" --train_script start_grpo_npu.sh
RC=$?

if [ "$RC" -ne 0 ]; then
    # 训练失败: 给上传进程短暂宽限期冲刷已有 checkpoint/日志, 再以原退出码失败
    FAIL_WAIT_MINUTES=${FAIL_WAIT_MINUTES:-5}
    echo "[FAIL] 训练退出码 $RC, 等待 ${FAIL_WAIT_MINUTES} 分钟上传已有产物后退出"
    sleep "${FAIL_WAIT_MINUTES}m"
    kill $PY1_PID 2>/dev/null
    exit "$RC"
fi

# 3. 正常结束处理: 等待最后一批 checkpoint 上传完成
WAIT_MINUTES=${WAIT_MINUTES:-30}
echo "[Step 3] 训练已结束，等待 ${WAIT_MINUTES} 分钟确保数据上传完成..."
sleep "${WAIT_MINUTES}m"

kill $PY1_PID 2>/dev/null
echo "[Done] 所有进程已处理完毕。"
exit 0
