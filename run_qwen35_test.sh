source /home/ma-user/cann8.1/Ascend/ascend-toolkit/set_env.sh
source /home/ma-user/cann8.1/Ascend/nnal/atb/set_env.sh
# source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:512
export DISABLE_VERSION_CHECK=1
export HCCL_EXEC_TIMEOUT=6120
export HCCL_CONNECT_TIMEOUT=6120
export ACL_DEVICE_SYNC_TIMEOUT=6120
NPROC_PER_NODE=8
NNODES=1
RANK=0 
# 日志目录
LOG_DIR="/home/ma-user/work/lyf/log_dir"
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/train_log_${TIMESTAMP}.txt"
ERROR_FILE="$LOG_DIR/train_error_${TIMESTAMP}.txt"

cd /home/ma-user/work/lyf/LlamaFactory-qwen35

ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $NNODES \
    --node_rank $RANK \
    --monitor-interval 1800 \
    src/train.py /home/ma-user/work/lyf/crash_1cam_2cls_3s_qwen35.yaml \
    2>&1 | tee $LOG_FILE
# 保存退出码
EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully. Log saved at $LOG_FILE"
else
    echo "❌ Training failed with exit code $EXIT_CODE. See log: $LOG_FILE"
    # 可选：保留最近失败日志为 latest_failed.log
    cp $LOG_FILE $LOG_DIR/latest_failed.log
fi

# 保留最后一次日志软链
ln -sf $LOG_FILE $LOG_DIR/latest.log

# python /home/ma-user/work/LLaMA-Factory/src/fake.py





# source /home/ma-user/cann8.1/Ascend/ascend-toolkit/set_env.sh
# source /home/ma-user/cann8.1/Ascend/nnal/atb/set_env.sh
# # source /usr/local/Ascend/ascend-toolkit/set_env.sh

# export DISABLE_VERSION_CHECK=1
# export HCCL_EXEC_TIMEOUT=4800  # 延长超时
# export HCCL_CONNECT_TIMEOUT=4800
# export HCCL_OP_TIMEOUT=3600         # HCCL 操作超时
# # export ASCEND_GLOBAL_EVENT_ENABLE=0
# NPROC_PER_NODE=8
# # NPROC_PER_NODE=1
# NNODES=1
# RANK=0 

# cd /home/ma-user/work/lyf/LLaMA-Factory

# # ASCEND_RT_VISIBLE_DEVICES=0 torchrun \
# ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
#     --nproc_per_node $NPROC_PER_NODE \
#     --nnodes $NNODES \
#     --node_rank $RANK \
#     src/train.py /home/ma-user/work/lyf/not_public_v3.yaml
    
# # python /home/ma-user/work/LLaMA-Factory/src/fake.py





# #!/bin/sh

# (
# set -x
# source /home/ma-user/cann8.1/Ascend/ascend-toolkit/set_env.sh
# source /home/ma-user/cann8.1/Ascend/nnal/atb/set_env.sh
# # source /usr/local/Ascend/ascend-toolkit/set_env.sh
# export DISABLE_VERSION_CHECK=1
# export HCCL_EXEC_TIMEOUT=4800 # 延长超时
# export HCCL_CONNECT_TIMEOUT=4800
# export HCCL_OP_TIMEOUT=3600 # HCCL 操作超时
# # export ASCEND_GLOBAL_EVENT_ENABLE=0
# NPROC_PER_NODE=8
# # NPROC_PER_NODE=1
# NNODES=1
# RANK=0
# cd /home/ma-user/work/lyf/LLaMA-Factory
# # ASCEND_RT_VISIBLE_DEVICES=0 torchrun \
# ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
#     --nproc_per_node $NPROC_PER_NODE \
#     --nnodes $NNODES \
#     --node_rank $RANK \
#     src/train.py /home/ma-user/work/lyf/debug_tmp.yaml
# ) 2>&1 | tee -a /home/ma-user/work/lyf/debug_1028.log