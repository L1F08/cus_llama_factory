#!/usr/bin/bash
# build_final_dataset.sh — 组装最终数据集（Path D, step 7+8）
#
# 剔除 → 分层划分 train/test → 只对 train oversample（防泄漏）。
#
# 用法：
#   bash build_final_dataset.sh
#   OVERSAMPLE_FACTOR=2 TEST_RATIO=0.1 bash build_final_dataset.sh
#
# 可选环境变量：
#   CORRECTED_JSON     apply_label_corrections 产出的修正大集
#   TRUE_HARD_JSON     true_hard_all.json（要 oversample 的真难例）
#   DROP_STEMS_FILE    要剔除的 stem/path 列表（如低帧视频），留空则不剔除
#   OUT_DIR            输出目录
#   TEST_RATIO         测试集比例           (默认 0.1)
#   OVERSAMPLE_FACTOR  train 侧难例总份数     (默认 3)
#   SEED               随机种子             (默认 42)

set -e

SCRIPT=/home/ma-user/work/lyf/build_final_dataset.py
BASE=/home/ma-user/work/lyf/data/0521_crash_1cam_2cls_train_all_3s
CORR=${BASE}/corrections_v2

CORRECTED_JSON=${CORRECTED_JSON:-${CORR}/train_all_label_corrected.json}
TRUE_HARD_JSON=${TRUE_HARD_JSON:-${CORR}/true_hard_all.json}
DROP_STEMS_FILE=${DROP_STEMS_FILE:-${BASE}/low_frame_stems.txt}   # filter_low_frame_videos.py 的产出
OUT_DIR=${OUT_DIR:-${BASE}/final_dataset}
TEST_RATIO=${TEST_RATIO:-0.1}
OVERSAMPLE_FACTOR=${OVERSAMPLE_FACTOR:-3}
SEED=${SEED:-42}

mkdir -p ${OUT_DIR}

# DROP 名单可选：不存在就不传
DROP_ARG=""
if [ -f "${DROP_STEMS_FILE}" ]; then
    DROP_ARG="--drop_stems_file ${DROP_STEMS_FILE}"
else
    echo "ℹ️  drop_stems_file 不存在(${DROP_STEMS_FILE})，跳过帧数剔除"
fi

echo "=================================================="
echo "  Build final dataset (split-then-oversample)"
echo "  corrected:   ${CORRECTED_JSON}"
echo "  true_hard:   ${TRUE_HARD_JSON}"
echo "  drop_stems:  ${DROP_STEMS_FILE} ${DROP_ARG:+(used)}"
echo "  out_dir:     ${OUT_DIR}"
echo "  test_ratio:  ${TEST_RATIO}   oversample: ${OVERSAMPLE_FACTOR}   seed: ${SEED}"
echo "=================================================="

if [ ! -f "${CORRECTED_JSON}" ]; then echo "❌ corrected_json not found"; exit 1; fi
if [ ! -f "${TRUE_HARD_JSON}" ]; then echo "❌ true_hard_json not found"; exit 1; fi

python ${SCRIPT} \
    --corrected_json ${CORRECTED_JSON} \
    --true_hard_json ${TRUE_HARD_JSON} \
    --out_dir ${OUT_DIR} \
    --test_ratio ${TEST_RATIO} \
    --oversample_factor ${OVERSAMPLE_FACTOR} \
    --seed ${SEED} \
    ${DROP_ARG}

echo ""
echo "train_final.json / test.json 已生成于: ${OUT_DIR}"
