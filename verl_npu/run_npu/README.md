# run_npu — verl GRPO 训练编排层 (Ascend NPU)

本目录是绕 verl (v0.9.0.dev, 官方已内置 Ascend NPU 支持) 搭建的训练编排层,
文件读入方式与训练启动方式与 `llm_ft_longtime` (LlamaFactory SFT 框架) **完全同构**,
可直接沿用相同的 ModelArts 提交习惯。

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
取输出中最后出现的合法标签作预测, 与 ground truth 精确匹配得 1 分, 否则 0 分;
另记录 `format`(输出是否恰为标签本身) 与 `pred` 供 tensorboard 分析。

## 超参调整

直接编辑 `grpo_para.yaml` (每行都是 verl 的 hydra 配置点, 等价于命令行
`key=value`)。基线取自 verl 官方 `examples/grpo_trainer/run_qwen3_vl_8b_fsdp.sh`
(NPU 分支) 与 `run_qwen3_5_2b_video_fsdp.sh` (视频), 按 qwendrive(Qwen3-VL-4B)
与二分类短答任务调整:

- `actor_rollout_ref.rollout.n=8` — GRPO 组大小; 响应仅数 token, 组可取大
- `data.max_response_length=16` — 无 CoT, 只输出标签
- `data.max_prompt_length=8192` — 视频 token 预算, 配合 `convert.fps/max_frames/max_pixels`
- `trainer.resume_mode=auto` — 断点续训: 任务重启后 `run.py` 先从 OBS 恢复
  最近完整 checkpoint 到 outputs/, verl 再自动续训 (上传进程会维护远端
  `latest_checkpointed_iteration.txt` 只指向已完整上传的 step)
- 显存吃紧时: 降 `convert.max_pixels`/`max_frames`, 或升
  `actor_rollout_ref.rollout.tensor_model_parallel_size`

临时改参不必改文件: `bash start_grpo_npu.sh` 之后的参数会透传, 或
`python launch_grpo.py --para grpo_para.yaml --dry_run` 先看最终命令。

## ModelArts 任务依赖字段 (aarch64 / 鲲鹏)

平台提交表单的 `pip_package` / `apt_package` 只能补轻量 Python 依赖;
**CANN / torch_npu / vLLM / vLLM-Ascend 必须由镜像提供** (vLLM-Ascend 要对着
CANN 源码编译, CANN 不是 pip 包)。

`pip_package` (空格分隔, 已逐包核验 aarch64 wheel 可用, 无需源码编译):

```
accelerate bytecode codetiming datasets dill hydra-core numpy<2.0.0 pandas<3 pyarrow>=15.0.0,<=24.0.0 peft>=0.15.2 pybind11 pylatexenc tensordict>=0.8.0,<=0.10.0,!=0.9.0 ray[default] torchdata einops qwen-vl-utils>=0.0.14 av hf_transfer tensorboard mathruler wandb TransferQueue==0.1.8 transformers==5.3.0 xgrammar==0.1.33
```

`apt_package`:

```
ffmpeg
```

aarch64 注意事项:

- **不要安装 `decord`**: PyPI 上 `decord` / `eva-decord` 均无 aarch64 wheel,
  会退化为源码编译并失败。`qwen-vl-utils>=0.0.14` 默认用 `av` (PyAV) 读视频,
  aarch64 wheel 齐备, 因此不装 decord 是正常路径。
- `triton-ascend==3.2.1` 需华为源 (`--extra-index-url
  https://triton-ascend.osinfra.cn/pypi/simple/`), 平台字段一般不支持自定义源,
  应由镜像预装 (官方 NPU 镜像自带), 故未列入上表。
- `moxing` 由 ModelArts 镜像内置, **不可**写入 pip_package (PyPI 同名包不是它)。
- `pandas<3` 是稳妥起见的上限: pandas 3.0 为破坏性大版本, verl 按 2.x 开发验证。

## 环境要求 (NPU 机器)

- CANN + torch_npu + vllm-ascend, 按 verl 官方安装脚本:
  `bash <verl根>/scripts/install_vllm_mcore_npu.sh` (FSDP 路线可
  `USE_MEGATRON=0`), 版本矩阵见 `docs/ascend_tutorial/get_start/install_guidance.rst`
- `pip install -r <verl根>/requirements-npu.txt && pip install -e <verl根>`
- conda 环境名默认 `verl_npu` (可用环境变量 `CONDA_ENV` 覆盖)

## 本地(无 NPU / 无 moxing)调试

- 数据转换可独立跑: `python data_convert.py --train_json xx.json --out_dir /tmp/out`
- 奖励自测: `python rewards/collision_risk.py`
- 查看最终训练命令: `python launch_grpo.py --para grpo_para.yaml --dry_run`
- `scripts/common/download.py` 对非 `obs://` 路径走本地拷贝, 无 moxing 也可跑通全流程
