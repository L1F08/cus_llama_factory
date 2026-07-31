#!/bin/bash
# verl GRPO 训练启动脚本 (Ascend NPU) — 对应 llm_ft_longtime/start_multi_node.sh 的角色。
#
# 与 SFT(torchrun) 的关键差异: verl 基于 Ray 调度。
#   单节点: 直接启动 main_ppo, verl 自行初始化本地 Ray;
#   多节点: rank0 起 ray head 并等待所有 worker 注册后启动训练,
#           其余节点 ray start --address 加入集群并保活。
#
# 依赖环境变量 (由 run.py 注入): PROJECT_ROOT, VERL_ROOT, LOG_DIR, GRPO_PARA

set -eo pipefail

echo "打印原生环境变量: $VC_TASK_INDEX  $VC_WORKER_HOSTS  $MA_NUM_GPUS"

# ---------- 基础路径 ----------
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "$0")" && pwd)}
VERL_ROOT=${VERL_ROOT:-$(dirname "$PROJECT_ROOT")}
LOG_DIR=${LOG_DIR:-"$PROJECT_ROOT/logs"}
GRPO_PARA=${GRPO_PARA:-"$PROJECT_ROOT/grpo_para.yaml"}
mkdir -p "$LOG_DIR"

# ---------- 节点拓扑 (ModelArts 原生变量) ----------
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}
export NODE_RANK="${VC_TASK_INDEX:-0}"
if [ -n "$VC_WORKER_HOSTS" ]; then
    NNODES=$(echo "$VC_WORKER_HOSTS" | awk -F',' '{print NF}')
    MASTER_HOST="${VC_WORKER_HOSTS%%,*}"
    MASTER_ADDR=$(getent hosts "$MASTER_HOST" | awk '{ print $1 }')
else
    NNODES=${NNODES:-1}
    MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
fi
RAY_PORT=${RAY_PORT:-6766}
echo "拓扑: NNODES=$NNODES NODE_RANK=$NODE_RANK MASTER_ADDR=$MASTER_ADDR"

# ---------- Ascend 环境 (CANN 8.3.RC2 + vllm-ascend 0.11.0rc3 组合) ----------
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# 对齐 MindSpeed-MM verl_plugin qwen3vl 示例 (ray_start.sh) 的环境变量
export VLLM_USE_V1=1
export USE_OPTIMIZED_MODEL=0          # RLHF 场景禁用 vllm-ascend 优化模型
export VLLM_ASCEND_ENABLE_NZ=0
export HCCL_BUFFSIZE=300
export TASK_QUEUE_ENABLE=1
export COMBINED_ENABLE=1
export PYTORCH_NPU_ALLOC_CONF="garbage_collection_threshold:0.85"
export MULTI_STREAM_MEMORY_REUSE=1
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
ulimit -n 32768 2>/dev/null || true

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
# 防止 Ray 覆盖 ASCEND_RT_VISIBLE_DEVICES (verl ray_utils 识别此变量)
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1

# HCCL 超时 (沿用 SFT 脚本的长超时, 覆盖节点间下载/转换进度差)
export HCCL_WHITELIST_DISABLE=1
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=10800
export ACL_DEVICE_SYNC_TIMEOUT=10800

# tensorboard 输出目录 (trainer.logger 含 tensorboard 时生效)
export TENSORBOARD_DIR="$LOG_DIR/tensorboard"
mkdir -p "$TENSORBOARD_DIR"

cd "$VERL_ROOT"

# ---------- 启动 ----------
# 多节点时 verl 需连接已有 ray 集群: 此 verl 版本 ray_init 无 address 键,
# 通过 hydra 追加 +ray_kwargs.ray_init.address=auto 接入 (单节点自起本地 ray)
launch_train() {
    local extra=()
    if [ "$NNODES" -gt 1 ]; then
        extra+=("+ray_kwargs.ray_init.address=auto")
    fi
    python3 "$PROJECT_ROOT/launch_grpo.py" --para "$GRPO_PARA" "${extra[@]}" \
        2>&1 | tee -a "$LOG_DIR/full_train.log"
}

if [ "$NNODES" -le 1 ]; then
    echo "[单节点] 直接启动 verl GRPO 训练"
    ret=0
    launch_train || ret=$?
    exit $ret
fi

if [ "$NODE_RANK" = "0" ]; then
    echo "[多节点-head] 启动 ray head: $MASTER_ADDR:$RAY_PORT"
    ray stop --force >/dev/null 2>&1 || true
    ray start --head --port "$RAY_PORT" \
        --node-ip-address "$MASTER_ADDR" \
        --resources "{\"NPU\": $NPROC_PER_NODE}"

    echo "[多节点-head] 等待 $NNODES 个节点全部注册..."
    # 每节点在此之前还要跑 run.py (下载模型/视频数据 + 转 parquet), 节点间
    # 进度差可能很大, 等待窗口对齐 HCCL_CONNECT_TIMEOUT (2h)
    python3 - "$NNODES" <<'PYEOF'
import sys, time
import ray
ray.init(address="auto")
want = int(sys.argv[1])
for _ in range(1440):         # 1440 x 5s = 2h
    alive = [n for n in ray.nodes() if n["Alive"]]
    print(f"[ray] 当前节点数 {len(alive)}/{want}")
    if len(alive) >= want:
        sys.exit(0)
    time.sleep(5)
print("[ray] 等待 worker 注册超时", file=sys.stderr)
sys.exit(1)
PYEOF

    echo "[多节点-head] 集群就绪, 启动训练"
    # 显式捕获退出码: set -e 下直接调用会在失败时跳过 ray stop 清理
    ret=0
    launch_train || ret=$?
    ray stop --force || true
    exit $ret
else
    echo "[多节点-worker $NODE_RANK] 加入 ray 集群 $MASTER_ADDR:$RAY_PORT"
    ray stop --force >/dev/null 2>&1 || true
    # head 可能尚未就绪 (还在跑下载/数据转换), 重试窗口对齐 HCCL_CONNECT_TIMEOUT (2h)
    joined=0
    for i in $(seq 1 1440); do
        if ray start --address "$MASTER_ADDR:$RAY_PORT" \
            --resources "{\"NPU\": $NPROC_PER_NODE}"; then
            joined=1
            break
        fi
        echo "[worker] 第 $i 次加入失败, 5s 后重试..."
        sleep 5
    done
    if [ "$joined" -ne 1 ]; then
        echo "[多节点-worker $NODE_RANK] 加入 ray 集群失败 ($MASTER_ADDR:$RAY_PORT), 退出" >&2
        exit 1
    fi

    echo "[多节点-worker $NODE_RANK] 已加入, 保活直到 head 退出"
    while ray health-check --address "$MASTER_ADDR:$RAY_PORT" >/dev/null 2>&1; do
        sleep 30
    done
    echo "[多节点-worker $NODE_RANK] head 已退出, worker 结束"
    ray stop --force || true
    exit 0
fi
