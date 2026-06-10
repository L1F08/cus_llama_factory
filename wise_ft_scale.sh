#!/usr/bin/bash
# wise_ft_scale.sh — WiSE-FT 式 LoRA 插值（robust fine-tuning，零训练）
#
# 把 LoRA 贡献按 α 缩放（= 向基座插值），merger(modules_to_save) 显式与基座插值。
# 每个 α 产出一个标准 adapter 目录 → 接现有 merge_qwen35.sh + 推理 + eval。
#
# 用法：
#   bash wise_ft_scale.sh
#   ALPHAS="0.9 0.8" ADAPTER_DIR=/path/to/ckpt bash wise_ft_scale.sh
#
# 选 α 的标准：Test1 P/R ≥ 0.98 约束下，Recall@Test2 最高。
# 预期甜点区间 0.8 ~ 0.95（α 太小会快速丢任务能力——基座对本任务零先验）。

set -e

SCRIPT=/home/ma-user/work/lyf/wise_ft_scale.py

# 要缩放的 adapter（Exp6 或 Exp7 的最终 checkpoint）
ADAPTER_DIR=${ADAPTER_DIR:-/home/ma-user/work/lyf/outmodel/crash_1cam_2cls_3s_v2}   # !!! 改成实际 adapter/checkpoint 路径
BASE_MODEL=${BASE_MODEL:-/home/ma-user/work/lyf/model/Qwen3_5-9B}
ALPHAS=${ALPHAS:-"0.95 0.9 0.85 0.8 0.7"}
OUT_ROOT=${OUT_ROOT:-/home/ma-user/work/lyf/outmodel/wise_ft_out}

echo "=================================================="
echo "  WiSE-FT LoRA scaling"
echo "  adapter:  ${ADAPTER_DIR}"
echo "  base:     ${BASE_MODEL}"
echo "  alphas:   ${ALPHAS}"
echo "  out_root: ${OUT_ROOT}"
echo "=================================================="

if [ ! -d "${ADAPTER_DIR}" ]; then echo "❌ adapter_dir not found"; exit 1; fi
if [ ! -d "${BASE_MODEL}" ]; then echo "❌ base_model not found"; exit 1; fi

python ${SCRIPT} \
    --adapter_dir ${ADAPTER_DIR} \
    --base_model ${BASE_MODEL} \
    --alphas ${ALPHAS} \
    --out_root ${OUT_ROOT}

echo ""
echo "下一步：对每个 ${OUT_ROOT}/alpha_*/ 跑 merge_qwen35.sh（adapter 路径指向该目录），"
echo "然后推理 + eval Test1/Test2，选 Test1≥98 下 Test2 最高的 α。"
