#!/usr/bin/bash
# 单卡单样本诊断脚本：验证 cls_head 推理 pipeline 的每一步
# 用法：bash debug_cls_head_inference.sh

####################### task ID #######################
taskid_for_sft=crash_1cam_2cls_train_3s_39k_0508-800_nothink   # 与训练任务一致

####################### NPU 环境 #######################
export ASCEND_RT_VISIBLE_DEVICES=0          # debug 只用 1 张卡
source /home/ma-user/cann8.1/Ascend/ascend-toolkit/set_env.sh
source /home/ma-user/cann8.1/Ascend/nnal/atb/set_env.sh

####################### 推理参数 #######################
# 注意力后端：NPU 上 fa2 有兼容性问题，默认走 sdpa
export INFER_ATTN_IMPL=sdpa

# Merge 后的模型路径（必须已经 cp 过 cls_head.bin + cls_head_meta.json）
merge_out_path=/home/ma-user/work/lyf/outmodel/${taskid_for_sft}-merge

# 用一个已知标签的样本来 sanity check（强烈建议先用训练集里的"高风险"样本）
# 可以从训练 manifest 里随便挑一条
test_video=/home/ma-user/work/lyf/data/0506_crash_1cam_2cls_test_39k_3s/sample_high_risk.mp4   # !!! 改成实际路径
# 也可以挑一条"安全"样本对照：
# test_video=/home/ma-user/work/lyf/data/.../sample_safe.mp4

# 视觉参数：必须严格匹配训练 yaml
video_max_pixels=589824                     # = 训练时的 video_max_pixels
video_fps=8.0                               # = 训练时的 video_fps

# 输出日志
log_dir=/home/ma-user/work/lyf/log_dir
mkdir -p ${log_dir}
log_file=${log_dir}/cls_head_debug_$(date +%Y%m%d_%H%M%S).log

####################### Run #######################
echo "=================================================="
echo "  cls_head Debug — single sample diagnostic"
echo "  merged_dir: ${merge_out_path}"
echo "  video:      ${test_video}"
echo "  log:        ${log_file}"
echo "=================================================="

python /home/ma-user/work/lyf/debug_cls_head_inference.py \
    --merged_dir ${merge_out_path} \
    --video ${test_video} \
    --video_max_pixels ${video_max_pixels} \
    --video_fps ${video_fps} \
    --device npu:0 \
    2>&1 | tee ${log_file}

echo ""
echo "Full log saved to: ${log_file}"
echo "Symlink: ${log_dir}/cls_head_debug_latest.log"
ln -sf ${log_file} ${log_dir}/cls_head_debug_latest.log
