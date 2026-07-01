#!/usr/bin/bash
# preview_video_sampling.sh — 预览模型实际看到的抽帧画面（判断 fps/分辨率够不够）
#
# 按 LlamaFactory 抽帧规则(floor(dur×fps)+linspace) + Qwen smart_resize，
# 对一条数据的三路视频产出每种 fps×分辨率 组合的接触表 PNG + token 预算表。
#
# 用法：
#   SAMPLE_JSON=/path/train.json INDEX=0 FPS="3 4 5 6" MAX_PIXELS="401408 589824" bash preview_video_sampling.sh
#   VIDEOS="/p/front.mp4 /p/left.mp4 /p/right.mp4" bash preview_video_sampling.sh

set -e

SCRIPT=/home/ma-user/work/lyf/preview_video_sampling.py
BASE=/home/ma-user/work/lyf/data/0626_crash_3cam_2cls_train_all_3s_-3_0

SAMPLE_JSON=${SAMPLE_JSON:-${BASE}/final_dataset/train_final.json}
INDEX=${INDEX:-0}
VIDEOS=${VIDEOS:-}                       # 直接给三路视频则优先用它
FPS=${FPS:-"3 4 5 6"}
MAX_PIXELS=${MAX_PIXELS:-"401408 589824"}
OUT_DIR=${OUT_DIR:-${BASE}/sampling_preview}
PER_ROW=${PER_ROW:-6}

echo "=================================================="
echo "  Preview model-seen frames"
echo "  fps:        ${FPS}"
echo "  max_pixels: ${MAX_PIXELS}"
echo "  out_dir:    ${OUT_DIR}"
echo "=================================================="

[ -f "${SCRIPT}" ] || { echo "❌ preview_video_sampling.py not found: ${SCRIPT}"; exit 1; }

if [ -n "${VIDEOS}" ]; then
    python ${SCRIPT} --videos ${VIDEOS} --fps ${FPS} --max_pixels ${MAX_PIXELS} \
        --out_dir ${OUT_DIR} --per_row ${PER_ROW}
else
    [ -f "${SAMPLE_JSON}" ] || { echo "❌ sample_json not found: ${SAMPLE_JSON}"; exit 1; }
    python ${SCRIPT} --sample_json ${SAMPLE_JSON} --index ${INDEX} \
        --fps ${FPS} --max_pixels ${MAX_PIXELS} --out_dir ${OUT_DIR} --per_row ${PER_ROW}
fi

echo ""
echo "把 ${OUT_DIR}/*.png 下载到本地看：接触表每帧标了时间戳，看关键时刻(触发前0.5-1s)采到没、目标清不清。"
