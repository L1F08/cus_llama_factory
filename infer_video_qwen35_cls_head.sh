#!/usr/bin/bash
####################### task ID #######################
#!!!
taskid_for_sft=crash_1cam_2cls_train_3s_39k_0509_cls_head-800_nothink #!!! **修改为你的模型ID**
#!!!

####################### NPU env #######################
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export HCCL_CONNECT_TIMEOUT=4800
# export HCCL_OP_TIMEOUT=3600

source /home/ma-user/cann8.1/Ascend/ascend-toolkit/set_env.sh
source /home/ma-user/cann8.1/Ascend/nnal/atb/set_env.sh

####################### model & data paths #######################
model_template=qwen3_5
#!!!
merge_out_path=/home/ma-user/work/lyf/outmodel/${taskid_for_sft}-merge #!!!
#!!!

#!!!
infer_dataset_path=/home/ma-user/work/lyf/data/0506_crash_1cam_2cls_test_39k_3s/test_0506_crash_1cam_2cls_test_39k_3s_front_with_ego_info_5256_3s_clipped_cleaned_dedup_4k.json
output_json=/home/ma-user/work/lyf/result1/${taskid_for_sft}/result_$(date +%m%d).json
#!!!

####################### 🔥 cls_head inference params (MUST match training) #######################
# These MUST match the training yaml — mismatch silently corrupts visual encoding
# (ViT pos_embed table is 2304 entries; >2304 patches/frame produces OOD hidden states)
export INFER_VIDEO_MAX_PIXELS=589824   # = 768×768 = exactly 2304 patches/frame
export INFER_VIDEO_FPS=8.0             # match training video_fps
export INFER_ATTN_IMPL=sdpa            # NPU FA2 had issues; sdpa is numerically equivalent

####################### sanity check ##############################
if [ ! -f "${merge_out_path}/cls_head.bin" ]; then
    echo "❌ ERROR: cls_head.bin not found in ${merge_out_path}"
    echo "   Did merge_qwen35.sh run successfully? cls_head.bin must be copied from checkpoint dir."
    exit 1
fi
if [ ! -f "${merge_out_path}/cls_head_meta.json" ]; then
    echo "❌ ERROR: cls_head_meta.json not found in ${merge_out_path}"
    exit 1
fi
echo "✅ cls_head files found in merged dir"

####################### infer task ################################
echo "=================================================="
echo "     Merge lora finished, It's time to infer!     "
echo "model: ${taskid_for_sft}"
echo "merged_dir: ${merge_out_path}"
echo "output: ${output_json}"
echo "devices: ${ASCEND_RT_VISIBLE_DEVICES}"
echo "video_max_pixels=${INFER_VIDEO_MAX_PIXELS}  video_fps=${INFER_VIDEO_FPS}  attn=${INFER_ATTN_IMPL}"
echo "=================================================="

mkdir -p $(dirname ${output_json})

export WORLD_SIZE=$(echo $ASCEND_RT_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

DISTRIBUTED_ARGS="
    --nproc_per_node $WORLD_SIZE \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port 6663
"

torchrun $DISTRIBUTED_ARGS /home/ma-user/work/lyf/infer_faster_qwen35_cls_head.py \
    --model_id $merge_out_path \
    --data_path ${infer_dataset_path} \
    --output_json ${output_json} \
    --image_folder ""
