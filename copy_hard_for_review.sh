#!/usr/bin/bash
# copy_hard_for_review.sh — 把 hard_fn / hard_fp 里"还没审过"的视频复制出来
#
# hard_fn.json 包含全部漏报（含你已经看过的高置信 label1_risk）。
# 本脚本扫描 suspect_videos_after_check 收集已审 stem，只复制未审的视频，
# 避免重复观看。
#
# 用法：
#   bash copy_hard_for_review.sh                 # 默认处理 hard_fn
#   HARD=fp bash copy_hard_for_review.sh         # 处理 hard_fp
#
# 可选环境变量：
#   HARD          fn | fp                (默认 fn)
#   HARD_JSON     直接指定 hard json     (覆盖 HARD)
#   REVIEWED_DIR  已审视频根目录          (递归扫描)
#   DEST_DIR      复制目标目录
#   LIMIT         >0 时只复制前 N 个      (默认 0=全部)
#
# 不需要 NPU / torch —— 纯标准库 Python。

set -e

SCRIPT=/home/ma-user/work/lyf/copy_hard_for_review.py

HARD=${HARD:-fn}                       # fn | fp
BASE=/home/ma-user/work/lyf/data/0521_crash_1cam_2cls_train_all_3s

# 默认输入：analyze_hard_examples.py 的输出目录里的 hard_fn.json / hard_fp.json
HARD_JSON=${HARD_JSON:-${BASE}/hard_examples/hard_${HARD}.json}   # !!! 改成实际路径

# 已审视频根目录（递归扫描 stem）
REVIEWED_DIR=${REVIEWED_DIR:-${BASE}/hard_examples/suspect_videos_after_check}

# 复制目标目录（新建）
DEST_DIR=${DEST_DIR:-${BASE}/hard_examples/hard_${HARD}_to_review}

LIMIT=${LIMIT:-0}

mkdir -p ${DEST_DIR}

echo "=================================================="
echo "  Copy hard_${HARD} (un-reviewed only) for review"
echo "  hard_json:     ${HARD_JSON}"
echo "  reviewed_dir:  ${REVIEWED_DIR}"
echo "  dest_dir:      ${DEST_DIR}"
echo "  limit:         ${LIMIT}"
echo "=================================================="

if [ ! -f "${SCRIPT}" ]; then
    echo "❌ ERROR: copy_hard_for_review.py not found: ${SCRIPT}"; exit 1
fi
if [ ! -f "${HARD_JSON}" ]; then
    echo "❌ ERROR: hard_json not found: ${HARD_JSON}"; exit 1
fi

python ${SCRIPT} \
    --hard_json ${HARD_JSON} \
    --reviewed_dir ${REVIEWED_DIR} \
    --dest_dir ${DEST_DIR} \
    --limit ${LIMIT}

echo ""
echo "未审视频已复制到: ${DEST_DIR}"
echo "去那个目录看视频，按 true/fake/not_sure/low_quality 分类即可。"
