#!/usr/bin/bash
# eval_calibration.sh — Post-hoc 校准评测脚本（路径 B）
#
# 对 prediction JSON 做 Platt / TS / Beta / Isotonic 校准对比，
# 并导出可部署的校准参数 JSON（用于把 best thresh 拉回 0.5）。
#
# 两种模式：
#   single:  在一个 prediction 文件上跑校准
#   batch:   在多个模型版本（v2/v3/v4）上批量跑、对比
#
# 用法：
#   bash eval_calibration.sh                       # 默认 single
#   TAG=v3 bash eval_calibration.sh                # 改 tag/输出命名
#   MODE=batch bash eval_calibration.sh            # 批量跑
#
# 可选环境变量：
#   MODE           single | batch       (默认 single)
#   TAG            输出命名标签         (默认 v3)
#   KFOLD          K-fold 折数          (默认 5)
#   TARGET_THRESH  部署目标阈值         (默认 0.5)
#   SEED           K-fold 随机种子      (默认 42)
#
# 不需要 NPU / torch — 纯 numpy + sklearn + scipy。

set -e

# ============= 模式选择 =============
MODE=${MODE:-single}              # single | batch
TAG=${TAG:-v3}                    # 输出命名（single 模式）
KFOLD=${KFOLD:-5}
TARGET_THRESH=${TARGET_THRESH:-0.5}
SEED=${SEED:-42}

####################### 共用路径 #######################
SCRIPT=/home/ma-user/work/lyf/eval_calibration.py   # 校准评测脚本

# GT JSONL —— 所有模型公用同一份测试集 GT
GT_JSONL=/home/ma-user/work/lyf/data/0506_crash_1cam_2cls_test_39k_3s/test_0506_crash_1cam_2cls_test_39k_3s_front_with_ego_info_5256_3s_clipped_cleaned_dedup_4k.jsonl   # !!! 必要时改路径

# 输出目录
out_dir=/home/ma-user/work/lyf/calibration_out
mkdir -p ${out_dir}

log_dir=/home/ma-user/work/lyf/log_dir
mkdir -p ${log_dir}

####################### 模式 1: single ========================
# 在单个 prediction 文件上跑校准对比
PRED_JSON=/home/ma-user/work/lyf/result1/crash_1cam_2cls_train_3s_39k_head_0511-800_nothink/result.json   # !!! 改成实际 prediction

# Holdout 模式（可选）：用单独的 val prediction 拟合 calibrator
# 留空则走默认的 K-fold 诚实评估
CALIB_PRED=                                       # e.g. /path/to/val_result.json
CALIB_GT=                                         # 留空 → 沿用 GT_JSONL

####################### 模式 2: batch ========================
# 在多个模型版本上跑，结果各自保存
# 注意：bash 关联数组需要 bash 4+
declare -A BATCH_PREDS=(
    ["v2_token_ce"]="/home/ma-user/work/lyf/result1/crash_1cam_2cls_train_3s_39k_0508-800_nothink/result_0509.json"
    ["v3_cls_head_a0.5_g2"]="/home/ma-user/work/lyf/result1/crash_1cam_2cls_train_3s_39k_head_0511-800_nothink/result.json"
    ["v4_cls_head_a0.6_g0"]="/home/ma-user/work/lyf/result1/crash_1cam_2cls_train_3s_39k_head_0512_gamma0_nothink/result.json"
)

####################### 运行函数 #######################
run_one() {
    local tag=$1
    local pred=$2
    local ts=$(date +%Y%m%d_%H%M%S)
    local log_file=${log_dir}/calibration_${tag}_${ts}.log
    local cal_json=${out_dir}/calibrator_${tag}.json

    echo "=================================================="
    echo "  Calibration eval — ${tag}"
    echo "  pred:    ${pred}"
    echo "  gt:      ${GT_JSONL}"
    if [ -n "${CALIB_PRED}" ]; then
        echo "  mode:    HOLDOUT (calib_pred=${CALIB_PRED})"
    else
        echo "  mode:    K-FOLD (K=${KFOLD})"
    fi
    echo "  target:  ${TARGET_THRESH}"
    echo "  out:     ${cal_json}"
    echo "  log:     ${log_file}"
    echo "=================================================="

    if [ ! -f "${pred}" ]; then
        echo "❌ ERROR: pred not found: ${pred}"
        return 1
    fi
    if [ ! -f "${GT_JSONL}" ]; then
        echo "❌ ERROR: gt not found: ${GT_JSONL}"
        return 1
    fi
    if [ ! -f "${SCRIPT}" ]; then
        echo "❌ ERROR: eval_calibration.py not found: ${SCRIPT}"
        return 1
    fi

    local extra_args=""
    if [ -n "${CALIB_PRED}" ]; then
        extra_args="--calib_pred ${CALIB_PRED}"
        if [ -n "${CALIB_GT}" ]; then
            extra_args="${extra_args} --calib_gt ${CALIB_GT}"
        fi
    fi

    python ${SCRIPT} \
        --pred ${pred} \
        --gt ${GT_JSONL} \
        --kfold ${KFOLD} \
        --target_thresh ${TARGET_THRESH} \
        --seed ${SEED} \
        --save_calibrator ${cal_json} \
        ${extra_args} \
        2>&1 | tee ${log_file}

    ln -sf ${log_file} ${log_dir}/calibration_${tag}_latest.log
}

####################### Run #######################
if [ "${MODE}" = "single" ]; then
    run_one "${TAG}" "${PRED_JSON}"

elif [ "${MODE}" = "batch" ]; then
    failed=()
    for tag in "${!BATCH_PREDS[@]}"; do
        pred="${BATCH_PREDS[$tag]}"
        if ! run_one "${tag}" "${pred}"; then
            failed+=("${tag}")
            echo "⚠️  ${tag} failed, continuing batch..."
        fi
        echo ""
    done

    echo "=================================================="
    echo "  Batch summary"
    echo "  Calibrator JSONs saved to: ${out_dir}/calibrator_*.json"
    echo "  Latest logs:               ${log_dir}/calibration_*_latest.log"
    if [ ${#failed[@]} -gt 0 ]; then
        echo "  Failed tags:               ${failed[*]}"
    else
        echo "  All ${#BATCH_PREDS[@]} models completed successfully."
    fi
    echo "=================================================="

else
    echo "❌ ERROR: unknown MODE='${MODE}'. Use MODE=single or MODE=batch."
    exit 1
fi
