#!/usr/bin/bash
# check_multicam_frames.sh — 排查三路视频帧数是否一致/偶数（多进程 decode）
#
# 针对 "Video features and video tokens do not match" 报错：
#   查每条样本三路(前/左/右)视频的真实解码帧数，标记
#   不一致 / 奇数(temporal_patch_size=2 要偶数) / 读不出。
#
# 用法：
#   bash check_multicam_frames.sh
#   DATASET_JSON=/path/train.json WORKERS=180 bash check_multicam_frames.sh

set -e

SCRIPT=/home/ma-user/work/lyf/check_multicam_frames.py
BASE=/home/ma-user/work/lyf/data/0626_crash_3cam_2cls_train_all_3s_-3_0

DATASET_JSON=${DATASET_JSON:-${BASE}/final_dataset/train_final.json}   # !!! 改成实际训练 json
WORKERS=${WORKERS:-180}
METHOD=${METHOD:-decode}
OUT_CSV=${OUT_CSV:-${BASE}/multicam_frames.csv}
BAD_STEMS=${BAD_STEMS:-${BASE}/multicam_bad_stems.txt}

echo "=================================================="
echo "  Check multi-cam frame consistency"
echo "  dataset:  ${DATASET_JSON}"
echo "  workers:  ${WORKERS}   method: ${METHOD}"
echo "  out_csv:  ${OUT_CSV}"
echo "  bad_stems:${BAD_STEMS}"
echo "=================================================="

[ -f "${SCRIPT}" ] || { echo "❌ check_multicam_frames.py not found: ${SCRIPT}"; exit 1; }
[ -f "${DATASET_JSON}" ] || { echo "❌ dataset_json not found: ${DATASET_JSON}"; exit 1; }

python ${SCRIPT} \
    --dataset_json ${DATASET_JSON} \
    --workers ${WORKERS} \
    --method ${METHOD} \
    --out_csv ${OUT_CSV} \
    --bad_stems ${BAD_STEMS}

echo ""
echo "异常样本可用 ${BAD_STEMS} 作 build_final_dataset 的 DROP_STEMS_FILE 剔除。"
