#!/usr/bin/bash
# triage_invisible_risk.sh — "不可见风险"正样本分诊 / 清洗（Exp10 流水线）
#
# 四种模式（两侧 × triage/apply）：
#   正样本（标高风险、判安全 → 疑似风险不可见）：
#     triage     建队列(P(安全)降序)+拷视频；   apply     剔 invisible/not_sure/low_quality
#   负样本（标安全、判高风险 → 疑似标错 或 硬负样本）：
#     triage_neg 建队列(P(高风险)降序)+拷视频； apply_neg 剔 mislabel_risk/not_sure/low_quality
#                ⚠️ 保留 hard_negative（overtake 类，精度护城河）
#
# 用法：
#   bash triage_invisible_risk.sh                          # 默认 triage（正样本）
#   MODE=triage_neg bash triage_invisible_risk.sh          # 负样本分诊
#   MODE=apply_neg REVIEW_DIR=<分拣目录> bash triage_invisible_risk.sh
#
# 可选环境变量：
#   MODE               triage | apply | triage_neg | apply_neg   (默认 triage)
#   == triage ==
#   PRED               Exp8 对全部"高风险"正样本的推理 JSON
#   PRED2              可选第二探针（如 Exp6 同批推理），both-safe 排最前
#   OUT_DIR            分诊输出目录（队列 csv + review_videos/）
#   P_SAFE_THRESHOLD   P(安全) ≥ 此值进队列      (默认 0.5)
#   LIMIT              >0 只拷贝队列前 N 个视频   (默认 0=全部)
#   == apply ==
#   TRAIN_JSON         -3..0 完整训练 manifest（含正负样本）
#   REVIEW_DIR         人工分拣后的目录（含 visible_risk/invisible/not_sure/low_quality）
#   OUT_JSON           清洗后训练集输出路径
#
# 纯 CPU / 标准库，无 NPU 依赖。

set -e

SCRIPT=/home/ma-user/work/lyf/triage_invisible_risk.py
BASE=/home/ma-user/work/lyf/data/0521_crash_1cam_2cls_train_all_3s   # !!! 按 -3..0 数据实际目录调整

MODE=${MODE:-triage}

# ---- triage ----
PRED=${PRED:-/home/ma-user/work/lyf/result1/exp8_pos_selfinfer/result.json}   # !!! 改成实际推理结果
PRED2=${PRED2:-}                                                              # 可选
OUT_DIR=${OUT_DIR:-${BASE}/triage_out}
P_SAFE_THRESHOLD=${P_SAFE_THRESHOLD:-0.5}
LIMIT=${LIMIT:-0}

# ---- apply ----
TRAIN_JSON=${TRAIN_JSON:-${BASE}/crash_1cam_2cls_3s_47k_-3_0_135_1_aug.json}  # !!! 改成实际 manifest
REVIEW_DIR=${REVIEW_DIR:-${OUT_DIR}/review_videos_after_check}
OUT_JSON=${OUT_JSON:-${BASE}/train_47k_-3_0_cleaned.json}
# 难例翻倍：人工确认难例(visible_risk/hard_negative)在训练集出现总份数（1=不翻倍, 2=翻倍, 3=三倍）
OVERSAMPLE_FACTOR=${OVERSAMPLE_FACTOR:-1}
# 难例来源 review 目录（空格分隔，可同时给正负两轮分拣目录）；空=用 REVIEW_DIR
HARD_REVIEW_DIRS=${HARD_REVIEW_DIRS:-}

# ---- 日志 ----
log_dir=/home/ma-user/work/lyf/log_dir
mkdir -p ${log_dir}
ts=$(date +%Y%m%d_%H%M%S)
log_file=${log_dir}/triage_${MODE}_${ts}.log

if [ ! -f "${SCRIPT}" ]; then echo "❌ triage_invisible_risk.py not found: ${SCRIPT}"; exit 1; fi

if [ "${MODE}" = "triage" ] || [ "${MODE}" = "triage_neg" ]; then
    if [ "${MODE}" = "triage_neg" ]; then SIDE_DESC="负样本（标安全、判高风险）"; SCORE="P(高风险)";
                                     else SIDE_DESC="正样本（标高风险、判安全）"; SCORE="P(安全)"; fi
    echo "=================================================="
    echo "  Triage [${MODE}] — ${SIDE_DESC}"
    echo "  pred:       ${PRED}"
    echo "  pred2:      ${PRED2:-（未用）}"
    echo "  out_dir:    ${OUT_DIR}"
    echo "  threshold:  ${SCORE} ≥ ${P_SAFE_THRESHOLD}"
    echo "  limit:      ${LIMIT}"
    echo "  log:        ${log_file}"
    echo "=================================================="
    if [ ! -f "${PRED}" ]; then echo "❌ pred not found: ${PRED}"; exit 1; fi

    PRED2_ARG=""
    if [ -n "${PRED2}" ]; then
        if [ ! -f "${PRED2}" ]; then echo "❌ pred2 not found: ${PRED2}"; exit 1; fi
        PRED2_ARG="--pred2 ${PRED2}"
    fi

    mkdir -p ${OUT_DIR}
    python -u ${SCRIPT} --mode ${MODE} \
        --pred ${PRED} ${PRED2_ARG} \
        --out_dir ${OUT_DIR} \
        --p_safe_threshold ${P_SAFE_THRESHOLD} \
        --limit ${LIMIT} \
        2>&1 | tee ${log_file}

    echo ""
    if [ "${MODE}" = "triage_neg" ]; then
        echo "人工分拣 ${OUT_DIR}/review_videos/ → hard_negative(留) / mislabel_risk / not_sure / low_quality(剔)"
        echo "分拣完成后：MODE=apply_neg REVIEW_DIR=<分拣目录> bash triage_invisible_risk.sh"
    else
        echo "人工分拣 ${OUT_DIR}/review_videos/ → visible_risk(留) / invisible / not_sure / low_quality(剔)"
        echo "分拣完成后：MODE=apply REVIEW_DIR=<分拣目录> bash triage_invisible_risk.sh"
    fi

elif [ "${MODE}" = "apply" ] || [ "${MODE}" = "apply_neg" ]; then
    echo "=================================================="
    echo "  Apply [${MODE}] review verdicts → cleaned set"
    echo "  train_json: ${TRAIN_JSON}"
    echo "  review_dir: ${REVIEW_DIR}"
    echo "  out_json:   ${OUT_JSON}"
    echo "  log:        ${log_file}"
    echo "=================================================="
    if [ ! -f "${TRAIN_JSON}" ]; then echo "❌ train_json not found: ${TRAIN_JSON}"; exit 1; fi
    if [ ! -d "${REVIEW_DIR}" ]; then echo "❌ review_dir not found: ${REVIEW_DIR}"; exit 1; fi

    OVS_ARG=""
    [ "${OVERSAMPLE_FACTOR}" -gt 1 ] && OVS_ARG="--oversample_factor ${OVERSAMPLE_FACTOR}"
    HARD_ARG=""
    [ -n "${HARD_REVIEW_DIRS}" ] && HARD_ARG="--hard_review_dirs ${HARD_REVIEW_DIRS}"

    python -u ${SCRIPT} --mode ${MODE} \
        --train_json ${TRAIN_JSON} \
        --review_dir ${REVIEW_DIR} \
        --out_json ${OUT_JSON} \
        ${OVS_ARG} ${HARD_ARG} \
        2>&1 | tee ${log_file}

    echo ""
    echo "清洗后集合: ${OUT_JSON}（仅剔除${OVS_ARG:+ + 难例×${OVERSAMPLE_FACTOR}}，无翻标签）"

else
    echo "❌ unknown MODE='${MODE}'. 用 triage | apply | triage_neg | apply_neg。"; exit 1
fi

ln -sf ${log_file} ${log_dir}/triage_${MODE}_latest.log
