#!/bin/bash
# ModelArts 训练环境固化脚本 — CANN 8.3.RC2 受限环境 (公司内网版本约束)
#
# 依据内部已验证组合 (Qwen3-VL GRPO 迁移经验):
#   基础镜像: CANN 8.3.RC2 + Python 3.11 + torch 2.7.1 + torch_npu 2.7.1
#   源码安装: vllm v0.11.0 / vllm-ascend v0.11.0rc3 / verl 7df2afb
#             / transformers 7a833d1 / MindSpeed-MM verl_plugin (Qwen3vl)
#
# 用法: 在制作镜像或任务启动脚本中执行
#   bash setup_env_ma.sh /path/to/code_dir
# code_dir 下应包含(或将由本脚本克隆) vllm / vllm-ascend / transformers / MindSpeed-MM;
# verl 使用本仓库自带的 vendored 副本 (已固定 7df2afb 并含 PR#4030 修复)。

set -eo pipefail

CODE_DIR=${1:-${MA_JOB_DIR:-$(pwd)}}
RUN_NPU_DIR=$(cd "$(dirname "$0")" && pwd)
VERL_DIR=$(dirname "$RUN_NPU_DIR")          # 本仓库 vendored verl (7df2afb + 补丁)
cd "$CODE_DIR"

# ---------- 0. 内网 pip 源 (按公司环境修改) ----------
# pip config set global.index-url "http://<内网源>/simple"
TRUSTED_HOST=${TRUSTED_HOST:-}              # 内网源 ip, 外网环境留空
DEFAULT_TIMEOUT=${DEFAULT_TIMEOUT:-100}
RETRIES=${RETRIES:-6}
PIP_OPTS="--default-timeout=$DEFAULT_TIMEOUT --retries=$RETRIES"
[ -n "$TRUSTED_HOST" ] && PIP_OPTS="$PIP_OPTS --trusted-host $TRUSTED_HOST"

echo "===== 0. 环境自检"
pip list 2>/dev/null | grep -iE "^(torch|torch-npu|vllm|verl|transformers|mindspeed)" || true
# 已装过 verl / mindspeed-mm 的环境务必卸载, 否则有兼容性问题
pip uninstall -y verl mindspeed-mm 2>/dev/null || true

echo "===== 1. 基础依赖"
pip install cmake==3.26.4 $PIP_OPTS
pip install torch==2.7.1 torchvision==0.22.1 $PIP_OPTS   # torch/torchvision 版本绑定
pip install torch_npu==2.7.1 $PIP_OPTS                    # 与 torch 版本绑定
pip install pybind11 $PIP_OPTS

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

echo "===== 2. vllm v0.11.0 (源码, CPU-only 构建)"
if [ ! -d vllm ]; then
    git clone https://gitcode.com/GitHub_Trending/vl/vllm.git vllm
fi
cd vllm && git checkout v0.11.0
pip install -r requirements/build.txt $PIP_OPTS
VLLM_TARGET_DEVICE=empty pip install -v -e . $PIP_OPTS
cd ..

echo "===== 3. vllm-ascend v0.11.0rc3 (源码编译自定义算子)"
# MOE 模型需 >= v0.11.0rc2; FLASHCOMM1 报错需 v0.11.0rc3 (vllm-ascend issue #4535)
if [ ! -d vllm-ascend ]; then
    git clone https://gitee.com/mirrors/vllm-ascend.git vllm-ascend
fi
cd vllm-ascend && git checkout v0.11.0rc3
pip install -r requirements.txt $PIP_OPTS
# 代码仓 requirements 可能把 torch 覆盖成 2.8.x, 后面第 6 步会重装回 2.7.1
# torch_npu 建议 Q3 商发版 (仅 ascend pypi 源有):
pip install torch-npu==2.7.1.dev20250724 -i https://mirrors.huaweicloud.com/ascend/repos/pypi || \
    echo "[WARN] ascend pypi 不可达, 沿用已装 torch_npu==2.7.1"
pip install -v -e . $PIP_OPTS
cd ..

echo "===== 4. verl (本仓库 vendored 7df2afb, 源码安装)"
export LD_LIBRARY_PATH=$(python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"):$LD_LIBRARY_PATH
VLLM_ASCEND_DIR=$(python3 -c "import vllm_ascend, os; print(os.path.dirname(vllm_ascend.__file__))" 2>/dev/null || true)
[ -n "$VLLM_ASCEND_DIR" ] && export LD_LIBRARY_PATH=$VLLM_ASCEND_DIR:$VLLM_ASCEND_DIR/lib64:$LD_LIBRARY_PATH
cd "$VERL_DIR"
export PYTHONPATH=$VERL_DIR:$PYTHONPATH
pip install -r requirements.txt $PIP_OPTS
pip install -v -e . $PIP_OPTS
cd "$CODE_DIR"

echo "===== 5. transformers @ 7a833d1"
if [ ! -d transformers ]; then
    git clone https://gitcode.com/GitHub_Trending/tra/transformers.git transformers
fi
cd transformers && git checkout 7a833d1ccd41673030c85107f65f454c0c3222f5
pip install '.[torch]' $PIP_OPTS
cd ..

echo "===== 6. 三方库 + 重装 torch (防止被上述步骤覆盖)"
pip install qwen-vl-utils==0.0.11 mathruler viztracer uvloop==0.21.0 \
    setuptools==80.9.0 cloudpickle==3.1.2 $PIP_OPTS
pip install torch==2.7.1 torchvision==0.22.1 $PIP_OPTS
pip install torch-npu==2.7.1.dev20250724 -i https://mirrors.huaweicloud.com/ascend/repos/pypi || \
    pip install torch_npu==2.7.1 $PIP_OPTS

echo "===== 7. MindSpeed-MM verl_plugin (Qwen3-VL NPU 适配)"
if [ ! -d MindSpeed-MM ]; then
    git clone https://gitcode.com/Ascend/MindSpeed-MM.git MindSpeed-MM
fi
cd MindSpeed-MM/verl_plugin
export MODEL_SELECT="Qwen3vl"
export VERL_PATH=$VERL_DIR
pip install -v -e . $PIP_OPTS
cd "$CODE_DIR"

echo "===== 8. 版本终检"
pip list 2>/dev/null | grep -iE "^(torch|torch-npu|vllm|vllm-ascend|verl|transformers|qwen-vl-utils|mindspeed)" || true
python3 -c "import torch, torch_npu; print('torch', torch.__version__, '| npu ok:', torch.npu.is_available())" || true
echo "[DONE] 环境配置完成"
