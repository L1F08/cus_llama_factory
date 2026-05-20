#!/usr/bin/bash
# cls_head debug 脚本 —— 支持两种模式：
#   single:  单样本 7 步深度诊断（验证 cls_head 加载、chat template、hidden state 位置）
#   batch:   小批量评测（看 N 个样本的 P 分布、accuracy、错例）
#
# 用法：
#   bash debug_cls_head_inference.sh          # 默认 single
#   MODE=batch bash debug_cls_head_inference.sh

set -e

# ============= 模式选择 =============
MODE=${MODE:-single}          # single | batch

####################### task ID #######################
taskid_for_sft=crash_1cam_2cls_train_3s_39k_head_0511-800_nothink   # !!! 改成你的训练任务ID

####################### NPU 环境 #######################
export ASCEND_RT_VISIBLE_DEVICES=0          # debug 只用 1 张卡
source /home/ma-user/cann8.1/Ascend/ascend-toolkit/set_env.sh
source /home/ma-user/cann8.1/Ascend/nnal/atb/set_env.sh

####################### 共用参数 #######################
export INFER_ATTN_IMPL=sdpa                  # NPU 上 fa2 兼容性问题，走 sdpa

merge_out_path=/home/ma-user/work/lyf/outmodel/${taskid_for_sft}-merge

# 视觉参数：必须严格匹配训练 yaml
video_max_pixels=589824                      # = 训练时的 video_max_pixels
video_fps=8.0                                # = 训练时的 video_fps

# 输出日志
log_dir=/home/ma-user/work/lyf/log_dir
mkdir -p ${log_dir}
log_file=${log_dir}/cls_head_debug_${MODE}_$(date +%Y%m%d_%H%M%S).log

####################### 模式 1: single =======================
# 单样本 7 步深度诊断 —— 验证 cls_head 加载、chat template、hidden state 位置等
test_video=/home/ma-user/work/lyf/data/0506_crash_1cam_2cls_test_39k_3s/sample_high_risk.mp4   # !!! 改成实际路径

####################### 模式 2: batch ========================
# 小批量评测 —— 看预测分布、accuracy、错例
# JSONL 文件，每行一条：{"video": "/path/to/x.mp4", "label": "高风险"}
batch_manifest=/home/ma-user/work/lyf/batch_debug.jsonl   # !!! 改成实际路径

####################### Run #######################
echo "=================================================="
echo "  cls_head Debug — mode: ${MODE}"
echo "  merged_dir: ${merge_out_path}"
if [ "${MODE}" = "single" ]; then
    echo "  video:      ${test_video}"
elif [ "${MODE}" = "batch" ]; then
    echo "  manifest:   ${batch_manifest}"
fi
echo "  video_max_pixels=${video_max_pixels}  video_fps=${video_fps}"
echo "  log:        ${log_file}"
echo "=================================================="

if [ "${MODE}" = "single" ]; then
    if [ ! -f "${test_video}" ]; then
        echo "❌ ERROR: test_video not found: ${test_video}"
        echo "   Please edit this script and set 'test_video' to a real video path."
        exit 1
    fi
    python /home/ma-user/work/lyf/debug_cls_head_inference.py \
        --merged_dir ${merge_out_path} \
        --video ${test_video} \
        --video_max_pixels ${video_max_pixels} \
        --video_fps ${video_fps} \
        --device npu:0 \
        2>&1 | tee ${log_file}

elif [ "${MODE}" = "batch" ]; then
    if [ ! -f "${batch_manifest}" ]; then
        echo "❌ ERROR: batch_manifest not found: ${batch_manifest}"
        echo "   Create a JSONL file with lines like:"
        echo "     {\"video\": \"/path/to/x.mp4\", \"label\": \"高风险\"}"
        echo "     {\"video\": \"/path/to/y.mp4\", \"label\": \"安全\"}"
        exit 1
    fi
    python /home/ma-user/work/lyf/debug_cls_head_inference.py \
        --merged_dir ${merge_out_path} \
        --manifest ${batch_manifest} \
        --video_max_pixels ${video_max_pixels} \
        --video_fps ${video_fps} \
        --device npu:0 \
        2>&1 | tee ${log_file}

else
    echo "❌ ERROR: unknown MODE='${MODE}'. Use MODE=single or MODE=batch."
    exit 1
fi

echo ""
echo "Full log saved to: ${log_file}"
echo "Symlink: ${log_dir}/cls_head_debug_latest.log"
ln -sf ${log_file} ${log_dir}/cls_head_debug_latest.log
