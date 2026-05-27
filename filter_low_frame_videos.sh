#!/usr/bin/bash
# filter_low_frame_videos.sh — 找出丢帧/坏视频，输出 drop 名单（Path D, step 6）
#
# 3s@~10fps ≈ 29-30 帧。低于 --min_frames 的视频 + 读不出的，写进 drop 名单，
# 供 build_final_dataset.sh 的 DROP_STEMS_FILE 使用。
#
# 用法：
#   bash filter_low_frame_videos.sh
#   MIN_FRAMES=29 bash filter_low_frame_videos.sh
#
# 注意：会用 cv2 逐个读 55k 视频的元数据，耗时若干分钟，跑一次即可。

set -e

SCRIPT=/home/ma-user/work/lyf/filter_low_frame_videos.py
BASE=/home/ma-user/work/lyf/data/0521_crash_1cam_2cls_train_all_3s

# 在修正后的大集上扫（也可换成原始 cleaned.json）
DATASET_JSON=${DATASET_JSON:-${BASE}/corrections_v2/train_all_label_corrected.json}
MIN_FRAMES=${MIN_FRAMES:-29}
OUT=${OUT:-${BASE}/low_frame_stems.txt}

echo "=================================================="
echo "  Filter low-frame videos"
echo "  dataset:     ${DATASET_JSON}"
echo "  min_frames:  ${MIN_FRAMES}"
echo "  out:         ${OUT}"
echo "=================================================="

if [ ! -f "${DATASET_JSON}" ]; then echo "❌ dataset_json not found: ${DATASET_JSON}"; exit 1; fi

python ${SCRIPT} \
    --dataset_json ${DATASET_JSON} \
    --min_frames ${MIN_FRAMES} \
    --out ${OUT}

echo ""
echo "drop 名单已写入: ${OUT}"
echo "下一步 build_final_dataset.sh 会自动用它（DROP_STEMS_FILE）。"
