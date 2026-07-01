#!/usr/bin/bash
# build_final_dataset.sh — 组装最终数据集（Path D, step 7+8）
#
# 精确数量控制：分别指定 train/test 的 高风险(1)/安全(0) 条数。
# test 先预留、train 从剩余里取，保证 train∩test=∅。
# 输出：train_final.json (ShareGPT) + test.json (推理输入) + test.jsonl (GT)。
#
# 用法：
#   bash build_final_dataset.sh
#   TRAIN_POS=18000 TRAIN_NEG=18000 TEST_POS=2000 TEST_NEG=2000 bash build_final_dataset.sh
#
# 可选环境变量：
#   CORRECTED_JSON     修正后大集
#   DROP_STEMS_FILE    要剔除的 stem/path 列表（低帧视频）
#   OUT_DIR            输出目录
#   TRAIN_POS/TRAIN_NEG  train 的 高风险/安全 条数
#   TEST_POS/TEST_NEG    test  的 高风险/安全 条数
#   TRUE_HARD_JSON     难例 json（可选，配合 oversample）
#   OVERSAMPLE_FACTOR  train 侧难例总份数（默认 1=不 oversample）
#   MAX_PIXELS         test.jsonl 视频段的 max_pixels（默认 336000）
#   SEED               随机种子（默认 42）

set -e

SCRIPT=/home/ma-user/work/lyf/build_final_dataset.py
BASE=/home/ma-user/work/lyf/data/0521_crash_1cam_2cls_train_all_3s
CORR=${BASE}/hard_examples/corrections_v2

CORRECTED_JSON=${CORRECTED_JSON:-${CORR}/train_all_label_corrected.json}
DROP_STEMS_FILE=${DROP_STEMS_FILE:-${BASE}/low_frame_stems.txt}
OUT_DIR=${OUT_DIR:-${BASE}/final_dataset}

TRAIN_POS=${TRAIN_POS:-18000}
TRAIN_NEG=${TRAIN_NEG:-18000}
TEST_POS=${TEST_POS:-2000}
TEST_NEG=${TEST_NEG:-2000}

TRUE_HARD_JSON=${TRUE_HARD_JSON:-${CORR}/true_hard_all.json}
OVERSAMPLE_FACTOR=${OVERSAMPLE_FACTOR:-1}
MAX_PIXELS=${MAX_PIXELS:-336000}
SEED=${SEED:-42}

mkdir -p ${OUT_DIR}

DROP_ARG=""
if [ -f "${DROP_STEMS_FILE}" ]; then
    DROP_ARG="--drop_stems_file ${DROP_STEMS_FILE}"
else
    echo "ℹ️  drop_stems_file 不存在(${DROP_STEMS_FILE})，跳过帧数剔除"
fi

HARD_ARG=""
if [ "${OVERSAMPLE_FACTOR}" -gt 1 ] && [ -f "${TRUE_HARD_JSON}" ]; then
    HARD_ARG="--true_hard_json ${TRUE_HARD_JSON} --oversample_factor ${OVERSAMPLE_FACTOR}"
fi

echo "=================================================="
echo "  Build final dataset (explicit counts)"
echo "  corrected:   ${CORRECTED_JSON}"
echo "  drop_stems:  ${DROP_STEMS_FILE} ${DROP_ARG:+(used)}"
echo "  out_dir:     ${OUT_DIR}"
echo "  train: 高风险 ${TRAIN_POS} + 安全 ${TRAIN_NEG}"
echo "  test:  高风险 ${TEST_POS} + 安全 ${TEST_NEG}"
echo "  oversample:  ${OVERSAMPLE_FACTOR} ${HARD_ARG:+(hard from ${TRUE_HARD_JSON})}"
echo "  max_pixels:  ${MAX_PIXELS}   seed: ${SEED}"
echo "=================================================="

if [ ! -f "${CORRECTED_JSON}" ]; then echo "❌ corrected_json not found"; exit 1; fi

python ${SCRIPT} \
    --corrected_json ${CORRECTED_JSON} \
    --out_dir ${OUT_DIR} \
    --train_pos ${TRAIN_POS} --train_neg ${TRAIN_NEG} \
    --test_pos ${TEST_POS} --test_neg ${TEST_NEG} \
    --max_pixels ${MAX_PIXELS} \
    --seed ${SEED} \
    ${DROP_ARG} ${HARD_ARG}

echo ""
echo "train_final.json / test.json / test.jsonl 已生成于: ${OUT_DIR}"
