#!/usr/bin/bash
####################### task ID #######################
#!!!
taskid_for_sft=crash_1cam_2cls_train_3s_39k_0508-800_nothink #!!! **修改为你的模型ID**
#!!!

####################### paramter #######################
# NPU 启动
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export HCCL_CONNECT_TIMEOUT=4800    # 默认 120，设为 1 小时
# export HCCL_OP_TIMEOUT=3600         # HCCL 操作超时

# ---- 批量推理参数 (logits_nat 模式) ----
# BATCH_SIZE: 每张 NPU 一次推理的样本数。
#   logits_nat 只取第一步 logits (max_new_tokens=1)，KV cache 只有 prefill，
#   显存压力远小于 thinking 模式 —— 910B 65GB 上 BATCH_SIZE 可以开到 8/12/16。
#   出现 OOM 时下调。
# PREPROCESS_WORKERS: CPU 端视频解码/tokenize 的并行线程数，与 NPU 推理流水线并行
export BATCH_SIZE=8
export PREPROCESS_WORKERS=6

source /home/ma-user/cann8.1/Ascend/ascend-toolkit/set_env.sh
source /home/ma-user/cann8.1/Ascend/nnal/atb/set_env.sh
# source /usr/local/Ascend/ascend-toolkit/set_env.sh

# model inout path # 换任务不需要修改
# model_in_path=/home/ma-user/work/lyf/model/Qwen3-VL-8B-Instruct
model_template=qwen3_5
# train_out_path=/home/ma-user/work/lyf/outmodel/${taskid_for_sft}
#!!!
merge_out_path=/home/ma-user/work/lyf/outmodel/${taskid_for_sft}-merge #!!!
#!!!
# /train-worker1-log/outmodel/qwen25-lora-cli-merge
# data input & infer output
# train_dataset=obstacle_in_intersection # **修改为你的训练数据**
#!!!
infer_dataset_path=/home/ma-user/work/lyf/data/0506_crash_1cam_2cls_test_39k_3s/test_0506_crash_1cam_2cls_test_39k_3s_front_with_ego_info_5256_3s_clipped_cleaned_dedup_4k.json
# /home/ma-user/work/lyf/test_pullover/data_json/infer_converted_data.json #!!! **修改为你的推理数据路径**
output_json=/home/ma-user/work/lyf/result1/${taskid_for_sft}/result_0509_logits_nat.json #!!!
# infer_dataset_path=/home/ma-user/work/lyf/data/0427_crash_1cam_2cls_test_v5_32k_3s/test_0427_crash_1cam_2cls_test_v5_32k_3s_front_with_ego_info_3931_deduplicated_3s_clipped_cleaned.json
# # # /home/ma-user/work/lyf/test_pullover/data_json/infer_converted_data.json #!!! **修改为你的推理数据路径**
# output_json=/home/ma-user/work/lyf/result1/${taskid_for_sft}/result_0430_logits_nat.json #!!!
#!!!
####################### infer task #######################
echo "=================================================="
echo "  logits_nat batch inference — output 安全/高风险 logits"
echo "model: ${taskid_for_sft} outpath: ${output_json}"
echo "Using devices: ${ASCEND_RT_VISIBLE_DEVICES}"
echo "BATCH_SIZE=${BATCH_SIZE} PREPROCESS_WORKERS=${PREPROCESS_WORKERS}"

# --- 新增：自动创建 output_json 所在的目录 ---
mkdir -p $(dirname ${output_json})
# ---------------------------------------------

# 获取 NPU 数量
export WORLD_SIZE=$(echo $ASCEND_RT_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

# 模型训练参数
# master_port 与 infer_video_qwen35.sh (6662) 错开，方便理论上同时占用不同 NPU 子集时不撞端口
DISTRIBUTED_ARGS="
    --nproc_per_node $WORLD_SIZE \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port 6663
"
torchrun $DISTRIBUTED_ARGS /home/ma-user/work/lyf/infer_faster_qwen35_logits_nat_batch.py \
    --model_id $merge_out_path \
    --data_path ${infer_dataset_path} \
    --output_json ${output_json} \
    --image_folder "" \


    # --local_rank -1
