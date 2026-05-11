#!/usr/bin/bash

export DISABLE_VERSION_CHECK=1

echo "=================================================="
echo "      Train lora finish, It's time to merge!      "
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
source /home/ma-user/cann8.1/Ascend/ascend-toolkit/set_env.sh
source /home/ma-user/cann8.1/Ascend/nnal/atb/set_env.sh

# 1. 标准 merge
llamafactory-cli export /home/ma-user/work/lyf/qwen3_5_lora_sft_merge.yaml

# 2. 从 yaml 解析路径（也可以直接写死）
ADAPTER_DIR=$(grep -E "^adapter_name_or_path:" /home/ma-user/work/lyf/qwen3_5_lora_sft_merge.yaml | awk '{print $2}')
EXPORT_DIR=$(grep -E "^export_dir:" /home/ma-user/work/lyf/qwen3_5_lora_sft_merge.yaml | awk '{print $2}')

# 3. 拷贝 cls_head 到 merged 目录
if [ -f "${ADAPTER_DIR}/cls_head.bin" ]; then
    cp "${ADAPTER_DIR}/cls_head.bin" "${EXPORT_DIR}/cls_head.bin"
    cp "${ADAPTER_DIR}/cls_head_meta.json" "${EXPORT_DIR}/cls_head_meta.json"
    echo "✅ Copied cls_head.bin + cls_head_meta.json to ${EXPORT_DIR}"
else
    echo "ℹ️  No cls_head found in ${ADAPTER_DIR} — assuming standard LoRA SFT."
fi