#!/usr/bin/bash
# filter_low_frame_videos.sh — 多进程统计帧数 + 排除丢帧视频（Path D, step 6）
#
# 默认用 DECODE 真实计数（逐帧 grab），不是容器元数据——元数据经常虚高，
# 导致真正丢帧的视频(实际<29但元数据写29/30)漏过过滤。decode 慢但准，
# 你有 192 CPU，多进程并行。
#
# 用法：
#   bash filter_low_frame_videos.sh
#   WORKERS=192 DROP_MAX_FRAMES=28 bash filter_low_frame_videos.sh
#
# 可选环境变量：
#   DATASET_JSON      要扫描的数据集
#   DROP_MAX_FRAMES   帧数 <= 此值的排除   (默认 28 → 保留 >=29)
#   WORKERS           进程数               (默认 192)
#   METHOD            decode | meta        (默认 decode=可靠)
#   COUNTS_CSV        每条视频帧数记录 CSV
#   OUT               drop 名单输出

set -e

SCRIPT=/home/ma-user/work/lyf/filter_low_frame_videos.py
BASE=/home/ma-user/work/lyf/data/0521_crash_1cam_2cls_train_all_3s

DATASET_JSON=${DATASET_JSON:-${BASE}/hard_examples/corrections_v2/train_all_label_corrected.json}
DROP_MAX_FRAMES=${DROP_MAX_FRAMES:-28}
WORKERS=${WORKERS:-192}
METHOD=${METHOD:-decode}
COUNTS_CSV=${COUNTS_CSV:-${BASE}/frame_counts.csv}
OUT=${OUT:-${BASE}/low_frame_stems.txt}

echo "=================================================="
echo "  Filter low-frame videos (multiprocessing)"
echo "  dataset:          ${DATASET_JSON}"
echo "  drop_max_frames:  <= ${DROP_MAX_FRAMES}"
echo "  workers:          ${WORKERS}"
echo "  method:           ${METHOD}"
echo "  counts_csv:       ${COUNTS_CSV}"
echo "  drop list (out):  ${OUT}"
echo "=================================================="

if [ ! -f "${DATASET_JSON}" ]; then echo "❌ dataset_json not found: ${DATASET_JSON}"; exit 1; fi

python ${SCRIPT} \
    --dataset_json ${DATASET_JSON} \
    --drop_max_frames ${DROP_MAX_FRAMES} \
    --workers ${WORKERS} \
    --method ${METHOD} \
    --counts_csv ${COUNTS_CSV} \
    --out ${OUT}

echo ""
echo "每条视频帧数: ${COUNTS_CSV}"
echo "drop 名单:    ${OUT}"
echo "下一步 build_final_dataset.sh 会自动用 ${OUT}（DROP_STEMS_FILE）。"
