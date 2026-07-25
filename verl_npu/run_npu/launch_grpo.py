#!/usr/bin/python3.10
# -*- coding: utf-8 -*-
"""把 grpo_para.yaml (扁平 hydra key -> value) 转成 CLI overrides 并启动 verl 训练。

用法:
    python launch_grpo.py --para grpo_para.yaml [--dry_run] [extra_override ...]

extra_override 会原样追加在生成的 overrides 之后 (后者优先级更高, 可临时改参)。
"""

import argparse
import os
import subprocess
import sys

import yaml


def format_value(v):
    """按 hydra override 语法格式化值。"""
    if isinstance(v, bool):
        return "True" if v else "False"
    if v is None:
        return "null"
    return str(v)


def build_overrides(para_path):
    with open(para_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    if not isinstance(params, dict):
        raise ValueError(f"{para_path} 应为扁平 key: value 映射")

    unresolved = [k for k, v in params.items() if v == "__AUTO__"]
    if unresolved:
        raise RuntimeError(
            f"以下参数仍为 __AUTO__, 请先由 run.py 填充或手工指定: {unresolved}")

    return [f"{k}={format_value(v)}" for k, v in params.items()]


def check_qwen_vl_utils():
    """verl 的视频数据通路依赖较新的 qwen_vl_utils kwargs; 版本过旧时数据集
    会被静默过滤为空 (异常被吞), 故启动前显式探测函数签名并快速失败。"""
    from inspect import signature
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as e:
        raise RuntimeError(
            f"缺少 qwen_vl_utils ({e}); 请 pip install 'qwen-vl-utils>=0.0.14'") from e
    missing = ({"image_patch_size", "return_video_metadata"}
               - set(signature(process_vision_info).parameters))
    if missing:
        raise RuntimeError(
            f"qwen_vl_utils 版本过旧, process_vision_info 缺少参数 {missing}; "
            f"请升级: pip install -U 'qwen-vl-utils>=0.0.14'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--para", type=str, required=True, help="grpo_para.yaml 路径")
    parser.add_argument("--dry_run", action="store_true", help="仅打印命令不执行")
    args, extra = parser.parse_known_args()

    if not args.dry_run:
        check_qwen_vl_utils()

    overrides = build_overrides(args.para)
    cmd = [sys.executable, "-m", "verl.trainer.main_ppo"] + overrides + extra

    print("[launch_grpo] 启动命令:")
    print("  " + " \\\n  ".join(cmd))
    if args.dry_run:
        return 0

    # 在 verl 仓库根目录下执行 (本文件位于 <verl根>/run_npu/)
    verl_root = os.environ.get(
        "VERL_ROOT",
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return subprocess.run(cmd, cwd=verl_root).returncode


if __name__ == "__main__":
    sys.exit(main())
