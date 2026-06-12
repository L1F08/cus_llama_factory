#!/usr/bin/bash
# triage_invisible_risk.sh — "不可见风险"正样本分诊 / 清洗（Exp10 流水线）
#
# 两种模式：
#   triage: 用模型自推理结果建审核队列（P(安全) 降序）+ 拷视频供人工分拣
#   apply:  人工分拣完成后，剔除 invisible/not_sure/low_quality，产出清洗后训练集
#
# 用法：
#   bash triage_invisible_risk.sh                      # 默认 triage
#   PRED2=/path/exp6_pred.json bash triage_invisible_risk.sh   # 双探针
#   MODE=apply bash triage_invisible_risk.sh
#
# 可选环境变量：
#   MODE               triage | apply        (默认 triage)
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

# ---- 日志 ----
log_dir=/home/ma-user/work/lyf/log_dir
mkdir -p ${log_dir}
ts=$(date +%Y%m%d_%H%M%S)
log_file=${log_dir}/triage_${MODE}_${ts}.log

if [ ! -f "${SCRIPT}" ]; then echo "❌ triage_invisible_risk.py not found: ${SCRIPT}"; exit 1; fi

if [ "${MODE}" = "triage" ]; then
    echo "=================================================="
    echo "  Triage invisible-risk positives"
    echo "  pred:       ${PRED}"
    echo "  pred2:      ${PRED2:-（未用）}"
    echo "  out_dir:    ${OUT_DIR}"
    echo "  threshold:  P(安全) ≥ ${P_SAFE_THRESHOLD}"
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
    python ${SCRIPT} --mode triage \
        --pred ${PRED} ${PRED2_ARG} \
        --out_dir ${OUT_DIR} \
        --p_safe_threshold ${P_SAFE_THRESHOLD} \
        --limit ${LIMIT} \
        2>&1 | tee ${log_file}

    echo ""
    echo "人工分拣：把 ${OUT_DIR}/review_videos/ 的视频分到"
    echo "  visible_risk/（留） invisible/ not_sure/ low_quality/（剔）"
    echo "分拣完成后：MODE=apply REVIEW_DIR=<分拣目录> bash triage_invisible_risk.sh"

elif [ "${MODE}" = "apply" ]; then
    echo "=================================================="
    echo "  Apply review verdicts → cleaned train set"
    echo "  train_json: ${TRAIN_JSON}"
    echo "  review_dir: ${REVIEW_DIR}"
    echo "  out_json:   ${OUT_JSON}"
    echo "  log:        ${log_file}"
    echo "=================================================="
    if [ ! -f "${TRAIN_JSON}" ]; then echo "❌ train_json not found: ${TRAIN_JSON}"; exit 1; fi
    if [ ! -d "${REVIEW_DIR}" ]; then echo "❌ review_dir not found: ${REVIEW_DIR}"; exit 1; fi

    python ${SCRIPT} --mode apply \
        --train_json ${TRAIN_JSON} \
        --review_dir ${REVIEW_DIR} \
        --out_json ${OUT_JSON} \
        2>&1 | tee ${log_file}

    echo ""
    echo "清洗后训练集: ${OUT_JSON} → 用它训 Exp10（batch 64，可试 num_train_epochs: 3）"

else
    echo "❌ unknown MODE='${MODE}'. Use MODE=triage or MODE=apply."; exit 1
fi

ln -sf ${log_file} ${log_dir}/triage_${MODE}_latest.log
