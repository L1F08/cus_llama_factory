# run_npu — verl GRPO 训练编排层 (Ascend NPU, CANN 8.3.RC2 受限环境)

本目录是绕 verl 搭建的训练编排层, 文件读入方式与训练启动方式与
`llm_ft_longtime` (LlamaFactory SFT 框架) **完全同构**, 可直接沿用相同的
ModelArts 提交习惯。

**版本基线** (受公司内网 CANN 8.3.RC2 约束, 采用内部已验证组合):
verl **7df2afb** (v0.7.0.dev, 本仓库已 vendor 并打 PR#4030 补丁) + vllm 0.11.0
+ vllm-ascend 0.11.0rc3 + torch/torch_npu 2.7.1 + qwen-vl-utils 0.0.11
+ MindSpeed-MM v2.3.0 verl_plugin (Qwen3vl)。完整矩阵见根目录 `UPSTREAM.md`,
环境安装见本目录 `setup_env_ma.sh`。

## 与 llm_ft_longtime 的文件对应关系

| llm_ft_longtime (SFT)            | run_npu (GRPO)                  | 说明 |
|----------------------------------|---------------------------------|------|
| `run.sh`                         | `run.sh`                        | 顶层入口: 后台增量上传 + 阻塞训练 |
| `run.py`                         | `run.py`                        | 编排: 下载 → 数据准备 → 回填参数 → 启动 |
| `train_para.yaml`                | `grpo_para.yaml`                | 训练超参 (改为 verl hydra key) |
| `start_multi_node.sh` (torchrun) | `start_grpo_npu.sh` (**Ray**)   | verl 用 Ray 调度, 不是 torchrun |
| —                                | `launch_grpo.py`                | yaml → hydra CLI overrides |
| —                                | `data_convert.py`               | LlamaFactory JSON → verl parquet |
| —                                | `rewards/collision_risk.py`     | GRPO 规则奖励 (高风险/安全 精确匹配) |
| `upload_check-point.py`          | `upload_check-point.py`         | 适配 verl 的 `global_step_N/` 目录 |
| `experiments/*/config.yaml`      | `experiments/*/config.yaml`     | schema 不变, 新增 `convert` 节 |
| `scripts/common/*`               | `scripts/common/*`              | moxing 下载/上传/平台日志 |

## 使用方法 (与 SFT 相同)

```bash
cd <verl根>/run_npu
bash run.sh experiments/qwendrive/collision_risk_grpo_01/
```

流程: `run.sh` 起后台 checkpoint 上传进程 → `run.py` 按实验 `config.yaml`
下载模型/数据 (moxing) → JSON 转 parquet → 从 OBS 恢复断点 checkpoint (若有)
→ 回填 `grpo_para.yaml` 的 `__AUTO__` 项并写出
`<实验目录>/grpo_para.resolved.yaml` (模板本身不被覆写, resolved 副本即本次
实验的超参存档) → 调 `start_grpo_npu.sh`。

### 多节点

`start_grpo_npu.sh` 自动读取 ModelArts 的 `VC_TASK_INDEX` / `VC_WORKER_HOSTS`:

- **rank0**: `ray start --head --resources '{"NPU": 8}'` → 等全部节点注册 → 启动
  `verl.trainer.main_ppo`;
- **其他节点**: `ray start --address head:6766` 加入集群, 保活到 head 退出。

单节点时跳过集群搭建, verl 自行初始化本地 Ray。

## 数据格式

输入沿用 LlamaFactory sharegpt JSON (即 `data_process/video_gen` 的产物):

```json
{"messages": [{"role": "user", "content": "<video>\n...提示词..."},
              {"role": "assistant", "content": "高风险"}],
 "videos": ["/path/to/cam_front.mp4"]}
```

`data_convert.py` 转成 verl `RLHFDataset` 的 parquet 行:
`data_source / prompt / videos / ability / reward_model.ground_truth / extra_info`,
其中 `videos` 条目携带 `fps / max_frames / max_pixels` (透传 `qwen_vl_utils`,
对应 SFT 里的 `video_fps` / `video_max_pixels`)。抽帧参数在实验
`config.yaml` 的 `convert` 节调整。

**注意**: JSON 里引用的视频路径必须在训练机可访问 — 用 `convert.video_root`
(相对路径拼根) 或 `convert.path_map` (前缀替换) 做重映射。

## 奖励函数

`rewards/collision_risk.py::compute_score` — 无 CoT 约束下的规则奖励:
剥掉首尾空白/标点后**严格精确匹配** 高风险/安全 (避免"不安全"等否定表述
因子串匹配骗分), 匹配 ground truth 得 1 分, 否则 0 分; 另记录 `format` 与
`pred` — 此 verl 版本会自动用字符串 `pred` 产出 maj@N 多数投票指标。

## 超参调整

直接编辑 `grpo_para.yaml` (每行都是 verl 的 hydra 配置点, 等价于命令行
`key=value`)。基线取自 MindSpeed-MM
`examples/verl_examples/qwen3vl/train_qwen3_vl_8b_grpo_full.sh`
(内部已验证的 NPU Qwen3-VL GRPO 配置), 按 qwendrive(Qwen3-VL-2B, 2.13B)
与二分类短答视频任务调整:

- `actor_rollout_ref.rollout.n=8` — GRPO 组大小; 响应仅数 token, 组可取大
- `data.max_response_length=16` — 无 CoT, 只输出标签
- `data.max_prompt_length=8192` — 视频 token 预算, 配合 `convert.fps/max_frames/max_pixels`
- `trainer.resume_mode=auto` — 断点续训: 任务重启后 `run.py` 先从 OBS 恢复
  最近完整 checkpoint 到 outputs/, verl 再自动续训 (上传进程会维护远端
  `latest_checkpointed_iteration.txt` 只指向已完整上传的 step)
- `data.filter_overlong_prompts=false` — **勿开启**: 此 verl 版本开启后会把
  每个视频完整解码两遍, 且解码失败的样本被静默丢弃; 视频统一 3s 长度确定,
  转换期已做存在性校验, 无需此过滤
- 显存吃紧时: 降 `convert.max_pixels`/`max_frames`, 或升
  `actor_rollout_ref.rollout.tensor_model_parallel_size`

临时改参不必改文件: `bash start_grpo_npu.sh` 之后的参数会透传, 或
`python launch_grpo.py --para grpo_para.yaml --dry_run` 先看最终命令。

## 环境安装 (CANN 8.3.RC2 受限环境)

**镜像基线**: CANN 8.3.RC2 + Python 3.11 + torch 2.7.1 + torch_npu 2.7.1
(内部分享镜像: https://ai.gitcode.com/Ascend-SACT/Qwen3-VL-30B-A3B-Instruct-GRPO)。

**训练环境**: vllm / vllm-ascend / transformers / MindSpeed-MM verl_plugin
均需**源码安装** (无法通过平台 pip_package 字段解决), 执行:

```bash
bash setup_env_ma.sh /path/to/code_dir
```

脚本固化了内部验证过的完整流程: 内网 pip 源与超时配置 → torch/torch_npu 2.7.1
→ vllm v0.11.0 (VLLM_TARGET_DEVICE=empty) → vllm-ascend v0.11.0rc3 (源码编译
自定义算子; 商发 torch_npu 2.7.1.dev20250724) → 本仓库 vendored verl (7df2afb,
已含 PR#4030 timing 补丁) → transformers@7a833d1 → qwen-vl-utils==0.0.11 等
三方库 → 重装 torch 防覆盖 → MindSpeed-MM v2.3.0 verl_plugin
(MODEL_SELECT=Qwen3vl, 对 verl 打 Qwen3-VL NPU 适配补丁)。

建议先在交互环境跑通 `setup_env_ma.sh` 后**制作镜像**, 任务提交时不再现装;
平台 `pip_package` 字段此时只需兜底轻量纯 Python 包 (镜像里已装则留空):

```
qwen-vl-utils==0.0.11 mathruler viztracer uvloop==0.21.0 setuptools==80.9.0 cloudpickle==3.1.2 tensorboard
```

`apt_package`: `ffmpeg` (视频解码兜底)。

aarch64 注意: **不要安装 `decord`** (PyPI 无 aarch64 wheel, 会源码编译失败);
`moxing` 由 ModelArts 镜像内置, 不可写入 pip_package。

## 已知问题与修复 (来自内部迁移经验)

- `DataProto.concat` 报 `Conflicting values for meta_info key 'timing'`
  (verl PR#4030) — 本仓库 vendored verl 已打补丁。
- vllm-ascend MOE 需 >= v0.11.0rc2; `VLLM_ASCEND_ENABLE_FLASHCOMM1=1` 报错
  需升 v0.11.0rc3 (vllm-ascend issue #4535)。
- vllm-ascend 的 requirements.txt 会把 torch 覆盖成 2.8.x — 装完须重装
  torch==2.7.1 (setup_env_ma.sh 已处理)。
- 环境里如已装过 verl / mindspeed-mm, 必须先卸载再装, 否则有兼容性问题。
- conda 环境名默认 `verl_npu` (可用环境变量 `CONDA_ENV` 覆盖)。

## 本地(无 NPU / 无 moxing)调试

- 数据转换可独立跑: `python data_convert.py --train_json xx.json --out_dir /tmp/out`
- 奖励自测: `python rewards/collision_risk.py`
- 查看最终训练命令: `python launch_grpo.py --para grpo_para.yaml --dry_run`
- `scripts/common/download.py` 对非 `obs://` 路径走本地拷贝, 无 moxing 也可跑通全流程
