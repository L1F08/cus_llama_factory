#!/usr/bin/bash

export DISABLE_VERSION_CHECK=1

echo "=================================================="
echo "      Train lora finish, It's time to merge!      "
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
source /home/ma-user/cann8.1/Ascend/ascend-toolkit/set_env.sh
source /home/ma-user/cann8.1/Ascend/nnal/atb/set_env.sh

llamafactory-cli export /home/ma-user/work/lyf/qwen3_5_lora_sft_merge.yaml