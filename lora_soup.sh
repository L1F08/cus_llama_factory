#!/usr/bin/bash
# lora_soup.sh — LoRA checkpoint/run 权重平均（model soup / SWA，零训练）
#
# 两种模式：
#   adapter: 直接平均 lora_A/B + merger，产出 adapter 目录（接现有 merge 流程；
#            还可继续叠 wise_ft_scale）。适合：同一 run 的多个 checkpoint。
#   merged:  在全权重空间做 soup（W_base+ΣwᵢΔWᵢ），产出可直接推理的完整模型。
#            适合：跨 run（如 Exp3+Exp6+Exp7 finals）。数学上无子空间问题。
#
# 用法：
#   # 同 run checkpoint soup（先用 CHECK=1 看 A 子空间是否对齐）
#   CHECK=1 ADAPTERS="ckptA ckptB" bash lora_soup.sh
#   MODE=adapter ADAPTERS="ckpt-2000 ckpt-2025 ckpt-2050" bash lora_soup.sh
#
#   # 跨 run soup（全权重空间，可加权）
#   MODE=merged ADAPTERS="exp3_ad exp6_ad exp7_ad" WEIGHTS="0.2 0.4 0.4" bash lora_soup.sh
#
# 经验：--check 的 mean cos(lora_A) > 0.5 → adapter 模式安全；否则用 merged。

set -e

SCRIPT=/home/ma-user/work/lyf/lora_soup.py

MODE=${MODE:-adapter}                # adapter | merged
ADAPTERS=${ADAPTERS:-""}             # 空格分隔的 adapter/checkpoint 目录列表 !!!
WEIGHTS=${WEIGHTS:-""}               # 可选；空=均匀
BASE_MODEL=${BASE_MODEL:-/home/ma-user/work/lyf/model/Qwen3_5-9B}
OUT_DIR=${OUT_DIR:-/home/ma-user/work/lyf/outmodel/soup_${MODE}}
CHECK=${CHECK:-0}

if [ -z "${ADAPTERS}" ]; then
    echo "❌ 请设置 ADAPTERS=\"dir1 dir2 ...\""; exit 1
fi

if [ "${CHECK}" = "1" ]; then
    python ${SCRIPT} --check --adapters ${ADAPTERS}
    exit 0
fi

echo "=================================================="
echo "  LoRA soup  (mode=${MODE})"
echo "  adapters: ${ADAPTERS}"
echo "  weights:  ${WEIGHTS:-uniform}"
echo "  out_dir:  ${OUT_DIR}"
echo "=================================================="

WEIGHT_ARG=""
if [ -n "${WEIGHTS}" ]; then WEIGHT_ARG="--weights ${WEIGHTS}"; fi

if [ "${MODE}" = "adapter" ]; then
    python ${SCRIPT} --mode adapter --adapters ${ADAPTERS} ${WEIGHT_ARG} --out_dir ${OUT_DIR}
    echo ""
    echo "产出 adapter：${OUT_DIR} → 接 merge_qwen35.sh；也可再叠 wise_ft_scale.sh。"
elif [ "${MODE}" = "merged" ]; then
    python ${SCRIPT} --mode merged --adapters ${ADAPTERS} ${WEIGHT_ARG} \
        --base_model ${BASE_MODEL} --out_dir ${OUT_DIR}
    echo ""
    echo "产出完整模型：${OUT_DIR} → 推理脚本 model_id 直接指向它（无需 merge）。"
else
    echo "❌ unknown MODE='${MODE}'"; exit 1
fi
