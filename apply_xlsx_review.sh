#!/usr/bin/bash
# apply_xlsx_review.sh — 用 XLSX 人工标注清洗 -3..0 风险池（Exp10）
#
# 读 triage 队列的 XLSX（real_label 列），打印 verdict 分布 + 探针有效性 +
# 场景/标注者分析，然后从训练 manifest 剔除非真风险样本（只剔除，不翻标签/换窗口）。
#
# 用法：
#   DRY=1 bash apply_xlsx_review.sh        # 先 dry_run 看分布，确认列与映射
#   bash apply_xlsx_review.sh              # 确认无误后正式产出
#
# 可选环境变量：
#   XLSX / TRAIN_JSON / OUT_JSON
#   VERDICT_COL（默认 real_label）/ STEM_COL（默认 stem）
#   SCENE_COL（默认 lyf_check，仅分析）/ DROP_LABELS（覆盖默认剔除集）
#   DRY=1 → 只分析不写出

set -e

SCRIPT=/home/ma-user/work/lyf/apply_xlsx_review.py
BASE=/home/ma-user/work/lyf/data/0604_crash_1cam_2cls_train_all_3s_-3_0

XLSX=${XLSX:-${BASE}/triage_out/review_labeled.xlsx}              # !!! 改成实际 xlsx
TRAIN_JSON=${TRAIN_JSON:-${BASE}/risk_pool_32k_-3_0.json}         # !!! 风险池 manifest（32k）
OUT_JSON=${OUT_JSON:-${BASE}/risk_pool_32k_-3_0_cleaned.json}
VERDICT_COL=${VERDICT_COL:-real_label}
STEM_COL=${STEM_COL:-stem}
SCENE_COL=${SCENE_COL:-}      # 默认不做场景交叉分析；如需可传某列名
KEEP_LABELS=${KEEP_LABELS:-visible_risk}      # 算作"人工确认难例"的 verdict 值
EXPORT_HARD=${EXPORT_HARD:-}                  # 设了就导出难例清单，供 build_final_dataset 翻倍

DRY_ARG=""
[ "${DRY:-0}" = "1" ] && DRY_ARG="--dry_run"
DROP_ARG=""
[ -n "${DROP_LABELS:-}" ] && DROP_ARG="--drop_labels ${DROP_LABELS}"

echo "=================================================="
echo "  Apply XLSX review → clean -3..0 risk pool"
echo "  xlsx:       ${XLSX}"
echo "  train_json: ${TRAIN_JSON}"
echo "  out_json:   ${OUT_JSON}  ${DRY_ARG:+(dry_run)}"
echo "  verdict:    ${VERDICT_COL}   stem: ${STEM_COL}   scene: ${SCENE_COL}"
echo "=================================================="

[ -f "${SCRIPT}" ] || { echo "❌ apply_xlsx_review.py not found: ${SCRIPT}"; exit 1; }
[ -f "${XLSX}" ] || { echo "❌ xlsx not found: ${XLSX}"; exit 1; }
[ -f "${TRAIN_JSON}" ] || { echo "❌ train_json not found: ${TRAIN_JSON}"; exit 1; }

SCENE_ARG=""
[ -n "${SCENE_COL}" ] && SCENE_ARG="--scene_col ${SCENE_COL}"
EXPORT_ARG=""
[ -n "${EXPORT_HARD}" ] && EXPORT_ARG="--export_hard ${EXPORT_HARD} --keep_labels ${KEEP_LABELS}"

python ${SCRIPT} \
    --xlsx ${XLSX} \
    --train_json ${TRAIN_JSON} \
    --out_json ${OUT_JSON} \
    --verdict_col ${VERDICT_COL} \
    --stem_col ${STEM_COL} \
    ${SCENE_ARG} ${DROP_ARG} ${EXPORT_ARG} ${DRY_ARG}

echo ""
echo "确认 verdict 分布/映射无误后（去掉 DRY=1）产出清洗池。"
echo "难例翻倍 → build_final_dataset：--corrected_json ${OUT_JSON} --true_hard_json ${EXPORT_HARD:-<难例json>} --oversample_factor K"
