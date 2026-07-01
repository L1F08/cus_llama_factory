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
VIDEO=${VIDEO:-}                         # 单个视频路径（优先级最高）
VIDEOS=${VIDEOS:-}                       # 多个视频路径（如三路）
FPS=${FPS:-"3 4 5 6"}
MAX_PIXELS=${MAX_PIXELS:-"401408 589824"}
OUT_DIR=${OUT_DIR:-${BASE}/sampling_preview}
PER_ROW=${PER_ROW:-6}
MAKE_VIDEO=${MAKE_VIDEO:-0}          # 1 = 额外导出 mp4
REALTIME=${REALTIME:-0}             # 1 = 原速(每帧1/fps秒,真实~3s)；0 = 慢放(每帧 HOLD_SEC 秒)
HOLD_SEC=${HOLD_SEC:-0.5}           # 慢放每帧显示秒数(REALTIME=0 时生效)

echo "=================================================="
echo "  Preview model-seen frames"
echo "  fps:        ${FPS}"
echo "  max_pixels: ${MAX_PIXELS}"
echo "  out_dir:    ${OUT_DIR}"
echo "=================================================="

[ -f "${SCRIPT}" ] || { echo "❌ preview_video_sampling.py not found: ${SCRIPT}"; exit 1; }

VID_ARG=""
[ "${MAKE_VIDEO}" = "1" ] && VID_ARG="--make_video --hold_sec ${HOLD_SEC}"
[ "${MAKE_VIDEO}" = "1" ] && [ "${REALTIME}" = "1" ] && VID_ARG="${VID_ARG} --realtime"

if [ -n "${VIDEO}" ]; then
    python ${SCRIPT} --video ${VIDEO} --fps ${FPS} --max_pixels ${MAX_PIXELS} \
        --out_dir ${OUT_DIR} --per_row ${PER_ROW} ${VID_ARG}
elif [ -n "${VIDEOS}" ]; then
    python ${SCRIPT} --videos ${VIDEOS} --fps ${FPS} --max_pixels ${MAX_PIXELS} \
        --out_dir ${OUT_DIR} --per_row ${PER_ROW} ${VID_ARG}
else
    [ -f "${SAMPLE_JSON}" ] || { echo "❌ sample_json not found: ${SAMPLE_JSON}"; exit 1; }
    python ${SCRIPT} --sample_json ${SAMPLE_JSON} --index ${INDEX} \
        --fps ${FPS} --max_pixels ${MAX_PIXELS} --out_dir ${OUT_DIR} --per_row ${PER_ROW} ${VID_ARG}
fi

echo ""
echo "PNG 接触表: ${OUT_DIR}/*.png（每帧标时间戳）"
[ "${MAKE_VIDEO}" = "1" ] && echo "mp4: ${OUT_DIR}/*_$([ "${REALTIME}" = "1" ] && echo realtime || echo slow).mp4"
