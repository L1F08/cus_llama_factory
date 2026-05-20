#!/usr/bin/bash
# 单样本 attention 可视化
# 用法：bash attention_viz_qwen35.sh

####################### task ID #######################
taskid_for_sft=crash_1cam_2cls_train_3s_39k_head_0511-800_nothink   # !!! 改成你的 task ID

####################### NPU 环境 #######################
export ASCEND_RT_VISIBLE_DEVICES=0
source /home/ma-user/cann8.1/Ascend/ascend-toolkit/set_env.sh
source /home/ma-user/cann8.1/Ascend/nnal/atb/set_env.sh

####################### 参数 #######################
merge_out_path=/home/ma-user/work/lyf/outmodel/${taskid_for_sft}-merge

# !!! 改成你想分析的具体视频路径
test_video=/home/ma-user/work/lyf/data/0506_crash_1cam_2cls_test_39k_3s/sample_fp_lateral.mp4

# 视觉参数必须匹配训练
video_max_pixels=589824
video_fps=8.0

# 注意力聚合层数（取最后 N 个 full_attention 层）
last_n_layers=4

# 输出目录（按视频名分子目录，方便对比多条样本）
viz_root=/home/ma-user/work/lyf/viz_out
video_name=$(basename ${test_video} .mp4)
output_dir=${viz_root}/${taskid_for_sft}/${video_name}

mkdir -p ${output_dir}

####################### Run #######################
echo "=================================================="
echo "  Attention Visualization"
echo "  model:      ${merge_out_path}"
echo "  video:      ${test_video}"
echo "  output:     ${output_dir}"
echo "  layers:     last ${last_n_layers} full-attn"
echo "=================================================="

if [ ! -d "${merge_out_path}" ]; then
    echo "❌ ERROR: merged model dir not found: ${merge_out_path}"
    exit 1
fi
if [ ! -f "${test_video}" ]; then
    echo "❌ ERROR: video not found: ${test_video}"
    exit 1
fi

log_file=${output_dir}/run.log
python /home/ma-user/work/lyf/attention_viz_qwen35.py \
    --merged_dir ${merge_out_path} \
    --video ${test_video} \
    --output_dir ${output_dir} \
    --video_max_pixels ${video_max_pixels} \
    --video_fps ${video_fps} \
    --last_n_layers ${last_n_layers} \
    --device npu:0 \
    2>&1 | tee ${log_file}

echo ""
echo "Output files:"
ls -la ${output_dir}/
echo ""
echo "Open ${output_dir}/overlay.png to view the heatmap grid."
