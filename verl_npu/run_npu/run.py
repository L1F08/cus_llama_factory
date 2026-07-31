#!/usr/bin/python3.10
# -*- coding: utf-8 -*-
"""verl GRPO 训练编排入口 — 与 llm_ft_longtime/run.py 同构。

流程:
    1. 读取实验目录 config.yaml (download/upload/convert 三节)
    2. moxing 下载模型与训练/测试 JSON
    3. LlamaFactory JSON -> verl parquet (data_convert.py)
    4. 用绝对路径回填 grpo_para.yaml (模型/数据/输出/reward/节点数)
    5. 启动训练脚本 (start_grpo_npu.sh: Ray 集群 + verl.trainer.main_ppo)
"""

import argparse
import os
import sys
import subprocess
import yaml
from datetime import datetime
from zoneinfo import ZoneInfo

# 添加模块搜索路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))   # <verl根>/run_npu
VERL_ROOT = os.path.dirname(PROJECT_ROOT)                    # <verl根>
sys.path.insert(0, PROJECT_ROOT)
print("harness 目录：", PROJECT_ROOT)
print("verl 仓库目录：", VERL_ROOT)

from scripts.common.download import download_all, restore_checkpoint
from scripts.common.platform_logger import PLAT_LOGGER
import data_convert


def get_json_file(data_cfg):
    train_json, test_json = None, None
    for item in data_cfg:
        if item["type"] == "train":
            train_json = item["local_path"]
            continue
        if item["type"] == "test":
            test_json = item["local_path"]
            continue
    return train_json, test_json


def detect_nnodes():
    """从 ModelArts 环境变量推断节点数, 本地默认 1。"""
    hosts = os.environ.get("VC_WORKER_HOSTS", "")
    if hosts:
        return len(hosts.split(","))
    return int(os.environ.get("NNODES", 1))


def convert_data(train_json, test_json, out_dir, convert_cfg):
    """JSON -> parquet。convert_cfg 来自实验 config.yaml 的 convert 节 (可缺省)。"""
    convert_cfg = convert_cfg or {}
    common = dict(
        data_source=convert_cfg.get("data_source", "collision_risk"),
        ability=convert_cfg.get("ability", "video_risk_cls"),
        video_root=convert_cfg.get("video_root"),
        path_map=[tuple(p.split(":", 1)) for p in convert_cfg.get("path_map", [])],
        fps=convert_cfg.get("fps", 8.0),
        max_frames=convert_cfg.get("max_frames", 24),
        max_pixels=convert_cfg.get("max_pixels", 589824),
        check_video_exists=convert_cfg.get("check_video_exists", True),
    )

    train_rows, stats = data_convert.convert_samples(
        data_convert._load_json(train_json), "train", **common)
    print(f"[STATS] train: {stats}")
    data_convert.assert_no_missing_videos(stats, "train")
    if not train_rows:
        raise RuntimeError(f"训练集转换后为空, 请检查: {train_json}")

    val_rows = []
    if test_json and os.path.exists(test_json):
        val_rows, val_stats = data_convert.convert_samples(
            data_convert._load_json(test_json), "val", **common)
        print(f"[STATS] val: {val_stats}")
        data_convert.assert_no_missing_videos(val_stats, "val")

    if not val_rows:
        val_size = convert_cfg.get("val_size", 0.05)
        print(f"[INFO] 测试集不可用(缺失或无标签), 从训练集切 {val_size} 作验证")
        train_rows, val_rows = data_convert.split_train_val(train_rows, val_size)

    train_parquet = os.path.join(out_dir, "train.parquet")
    val_parquet = os.path.join(out_dir, "val.parquet")
    data_convert.save_parquet(train_rows, train_parquet)
    data_convert.save_parquet(val_rows, val_parquet)
    return train_parquet, val_parquet


def patch_grpo_para(para_file, out_file, updates):
    """读取模板 grpo_para.yaml, 回填 __AUTO__ 项后写出 resolved 副本。

    不覆写模板本身: 模板是用户编辑面 (含注释与 __AUTO__ 标记), 覆写会丢失
    全部注释; resolved 副本同时留作本次实验的超参存档。
    """
    with open(para_file, "r", encoding="utf-8") as f:
        cfg_para = yaml.safe_load(f)

    for key, value in updates.items():
        cfg_para[key] = value

    with open(out_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_para, f, allow_unicode=True, sort_keys=False)
    print(f"[OK] 训练参数已写出: {out_file}")
    for k, v in updates.items():
        print(f"    {k} = {v}")


def main():
    PLAT_LOGGER.info("[Training Stage]init environment done")

    parser = argparse.ArgumentParser(description="verl GRPO 训练框架 (NPU)")
    parser.add_argument("--experiment_dir", type=str, required=True,
                        help="实验目录, 例如 experiments/qwendrive/collision_risk_grpo_01")
    parser.add_argument("--train_script", type=str, default="start_grpo_npu.sh",
                        help="训练启动脚本 (Ray + verl main_ppo)")
    args = parser.parse_args()

    exp_dir = os.path.abspath(args.experiment_dir)
    print("实验绝对路径：", exp_dir)
    if not os.path.exists(exp_dir):
        raise FileNotFoundError(f"实验目录不存在: {exp_dir}")

    # 加载配置
    config_path = os.path.join(exp_dir, "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    output_path = config["upload"]["check_point"]
    print("训练结果保存在：", output_path)
    print(f"[{datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')}] "
          f"开始启动实验: {exp_dir}")

    # 1. 下载模型、数据
    print("===== 1. 开始下载...")
    download_all(config.get("download", {}))
    PLAT_LOGGER.info("[Training Stage]data download done")

    # 2. JSON -> parquet
    train_json, test_json = get_json_file(config["download"]["data"])
    if train_json is None:
        raise RuntimeError(f"找不到训练json文件，请检查配置文件:{config}")
    train_json = os.path.abspath(train_json)
    test_json = os.path.abspath(test_json) if test_json else None

    parquet_dir = os.path.join(exp_dir, "data")
    os.makedirs(parquet_dir, exist_ok=True)
    train_parquet, val_parquet = convert_data(
        train_json, test_json, parquet_dir, config.get("convert"))
    print("===== 2. 训练数据准备完成！")

    # 3. 断点续训: 尝试从 OBS 恢复上次任务的 checkpoint
    outputs_dir = os.path.join(exp_dir, "outputs")
    logs_dir = os.path.join(exp_dir, "outputs", "logs")
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    restore_checkpoint(output_path, outputs_dir)

    # 4. 回填训练参数 (写 resolved 副本, 不动模板)
    model_path = os.path.abspath(os.path.join(
        PROJECT_ROOT, config["download"]["model"]["local_path"]))
    reward_path = os.path.join(PROJECT_ROOT, "rewards", "collision_risk.py")

    para_file = os.path.join(PROJECT_ROOT, "grpo_para.yaml")
    resolved_para = os.path.join(exp_dir, "grpo_para.resolved.yaml")
    patch_grpo_para(para_file, resolved_para, {
        "actor_rollout_ref.model.path": os.path.normpath(model_path),
        "data.train_files": train_parquet,
        "data.val_files": val_parquet,
        "trainer.default_local_dir": os.path.abspath(outputs_dir),
        "custom_reward_function.path": reward_path,   # verl 7df2afb: 顶层键
        "trainer.nnodes": detect_nnodes(),
    })
    print("===== 3. 训练参数文件处理完成！")

    PLAT_LOGGER.info("[Training Stage]start train prepare done")

    # 4. 启动训练
    train_script = os.path.join(PROJECT_ROOT, args.train_script) \
        if not os.path.isabs(args.train_script) else args.train_script
    if not os.access(train_script, os.X_OK):
        subprocess.run(["chmod", "+x", train_script])

    print("===== 4. 开始执行训练...")
    train_env = os.environ.copy()
    train_env["PROJECT_ROOT"] = PROJECT_ROOT
    train_env["VERL_ROOT"] = VERL_ROOT
    train_env["LOG_DIR"] = logs_dir
    train_env["GRPO_PARA"] = resolved_para

    subprocess.run(["bash", train_script], env=train_env, check=True)


if __name__ == "__main__":
    main()
