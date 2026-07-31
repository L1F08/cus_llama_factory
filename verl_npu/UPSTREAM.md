# 上游来源

本目录基于 verl 官方仓库的克隆, 移除了嵌套 .git 以便随本仓库一起提交:

- 仓库: https://github.com/volcengine/verl
- commit: **7df2afb936cd37b7b3a262edc119b2a57f070e3b** (2025-10-24, v0.7.0.dev,
  "[recipe] fix: Qwen3-vl moe model patch (#3878)")
- 本地新增: `run_npu/` 训练编排层
- 本地补丁: `verl/protocol.py` DataProto.concat 对 meta_info `timing` 键
  容忍 rank 间差异 (对应上游 PR #4030, 该 commit 尚未包含此修复)

## 为什么固定在 7df2afb (而非最新版)

公司内网环境锁定 **CANN 8.3.RC2**, 无法使用新版 verl 要求的 CANN 9.0.0 /
torch 2.9 / vllm 0.18 栈。7df2afb 是内部已验证的可用组合基线:

| 软件 | 版本 |
|---|---|
| CANN | 8.3.RC2 |
| Python | 3.11 |
| torch / torchvision | 2.7.1 / 0.22.1 |
| torch_npu | 2.7.1 (商发 Q3: 2.7.1.dev20250724) |
| vllm | v0.11.0 (源码, VLLM_TARGET_DEVICE=empty) |
| vllm-ascend | v0.11.0rc3 (源码编译; MOE 需 >=rc2, FLASHCOMM1 报错需 rc3) |
| transformers | 7a833d1ccd41673030c85107f65f454c0c3222f5 (源码) |
| qwen-vl-utils | 0.0.11 |
| MindSpeed-MM | v2.3.0 的 verl_plugin (MODEL_SELECT=Qwen3vl) |

环境安装脚本见 `run_npu/setup_env_ma.sh`。
如需同步上游: 以 7df2afb 为基线对比/重放上游变更, 并注意保留 protocol.py 补丁。
