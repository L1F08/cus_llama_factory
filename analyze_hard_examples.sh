#!/usr/bin/bash
# analyze_hard_examples.sh — 训练集难例挖掘 launcher（路径 D）
#
# 把"训练集推理结果"和"带标签训练集"对齐，算每条的 P 和难度分，
# 导出难例集（原训练格式，可直接喂回 LlamaFactory）+ 诊断报告。
#
# 用法：
#   bash analyze_hard_examples.sh
#   OVERSAMPLE_FACTOR=3 bash analyze_hard_examples.sh     # 顺便生成增强 manifest
#
# 可选环境变量（都有默认值）：
#   PRED              训练集推理结果 JSON
#   TRAIN             带标签训练集 JSON
#   OUT_DIR           输出目录
#   THRESHOLD         决策阈值                 (默认 0.5)
#   BORDERLINE_BAND   |P-0.5|<此 → borderline  (默认 0.15 → P∈0.35~0.65)
#   HIGH_CONF_BAND    |P-0.5|>=此 → 高置信      (默认 0.4  → P<0.1 或 >0.9)
#   OVERSAMPLE_FACTOR >1 时生成 原集 + hard×(f-1) 的增强训练集 (默认 0=不生成)
#   SUSPECT_VIDEO_DIR 把疑似标签噪声的视频复制到此目录，方便人工抽查
#
# 不需要 NPU / torch —— 纯 numpy-free Python（只用标准库）。

set -e

SCRIPT=/home/ma-user/work/lyf/analyze_hard_examples.py

####################### 输入路径 #######################
# 训练集转测试格式后的推理结果（id=视频路径, logits.{安全,高风险}）
PRED=${PRED:-/home/ma-user/work/lyf/result1/crash_1cam_2cls_train_3s_39k_0519-1025_nothink/result_0521_train_infer.json}   # !!! 改成实际路径

# 带标签训练集（messages[assistant].content = "0"安全 / "1"高风险, videos[0]=视频路径）
TRAIN=${TRAIN:-/home/ma-user/work/lyf/data/crash_1cam_2cls_train_3s_39k_with_ego_info.json}   # !!! 改成实际路径

####################### 参数 #######################
OUT_DIR=${OUT_DIR:-/home/ma-user/work/lyf/hard_examples}
THRESHOLD=${THRESHOLD:-0.5}
BORDERLINE_BAND=${BORDERLINE_BAND:-0.15}
HIGH_CONF_BAND=${HIGH_CONF_BAND:-0.4}
OVERSAMPLE_FACTOR=${OVERSAMPLE_FACTOR:-0}

# 疑似标签噪声视频的复制目标目录（在下面 mkdir 新建）
SUSPECT_VIDEO_DIR=${SUSPECT_VIDEO_DIR:-/home/ma-user/work/lyf/hard_examples/suspect_videos}

####################### 新建目录 #######################
mkdir -p ${OUT_DIR}
mkdir -p ${SUSPECT_VIDEO_DIR}        # 新建疑似噪声视频目录

####################### 日志 #######################
log_dir=/home/ma-user/work/lyf/log_dir
mkdir -p ${log_dir}
ts=$(date +%Y%m%d_%H%M%S)
log_file=${log_dir}/hard_examples_${ts}.log

####################### Run #######################
echo "=================================================="
echo "  Hard-example mining (Path D)"
echo "  pred:             ${PRED}"
echo "  train:            ${TRAIN}"
echo "  out_dir:          ${OUT_DIR}"
echo "  threshold:        ${THRESHOLD}"
echo "  borderline_band:  ${BORDERLINE_BAND}"
echo "  high_conf_band:   ${HIGH_CONF_BAND}"
echo "  oversample:       ${OVERSAMPLE_FACTOR}"
echo "  suspect_vid_dir:  ${SUSPECT_VIDEO_DIR}"
echo "  log:              ${log_file}"
echo "=================================================="

if [ ! -f "${SCRIPT}" ]; then
    echo "❌ ERROR: analyze_hard_examples.py not found: ${SCRIPT}"
    exit 1
fi
if [ ! -f "${PRED}" ]; then
    echo "❌ ERROR: pred not found: ${PRED}"
    exit 1
fi
if [ ! -f "${TRAIN}" ]; then
    echo "❌ ERROR: train not found: ${TRAIN}"
    exit 1
fi

python ${SCRIPT} \
    --pred ${PRED} \
    --train ${TRAIN} \
    --out_dir ${OUT_DIR} \
    --threshold ${THRESHOLD} \
    --borderline_band ${BORDERLINE_BAND} \
    --high_conf_band ${HIGH_CONF_BAND} \
    --oversample_factor ${OVERSAMPLE_FACTOR} \
    --suspect_video_dir ${SUSPECT_VIDEO_DIR} \
    2>&1 | tee ${log_file}

ln -sf ${log_file} ${log_dir}/hard_examples_latest.log

echo ""
echo "Full log: ${log_file}"
echo "Hard sets under: ${OUT_DIR}"
