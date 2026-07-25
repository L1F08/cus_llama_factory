#!/usr/bin/python3.10
# -*- coding: utf-8 -*-
"""check-point 增量上传 (与 llm_ft_longtime/upload_check-point.py 同构)。

与 SFT 版差异:
  1. verl 的 checkpoint 目录为 outputs/global_step_N/ (而非 checkpoint-N),
     平台"首个 loss 产出"标记按 global_step 匹配;
  2. verl 每次保存都会原地覆写 latest_checkpointed_iteration.txt (断点续训
     追踪文件)。该文件不能走一次性去重上传 (否则远端永远指向第一个 step),
     也不能直接实时上传 (可能指向尚未传完的 step) — 因此单独处理: 每轮结束后
     计算"已完整上传的最新 global_step", 幂等覆写远端追踪文件。
"""

import argparse
import os
import re
import shutil
import tempfile
import time
import yaml
from datetime import datetime
from zoneinfo import ZoneInfo

try:
    import moxing as mox
except ImportError:
    mox = None

from scripts.common.platform_logger import PLAT_LOGGER


def get_files(record_file, local_path):
    # 1. 读取已上传记录
    uploaded_files = set()
    if os.path.exists(record_file):
        with open(record_file, "r", encoding="utf-8") as f:
            uploaded_files = set(line.strip() for line in f if line.strip())

    # 2. 扫描当前目录下所有文件
    current_all_items = []
    now_time = time.time()
    STABLE_THRESHOLD = 60 * 8  # 稳定阈值：8 分钟内未修改的文件才允许上传

    for root, _, files in os.walk(local_path):
        for name in files:
            # 追踪文件是可变文件, 不走去重上传通路 (见模块 docstring), 单独处理
            if name == "latest_checkpointed_iteration.txt":
                continue
            full_path = os.path.join(root, name)
            try:
                file_stat = os.stat(full_path)
                mtime = file_stat.st_mtime
                is_exempt = ("trainer_log.jsonl" in name or "logs" in full_path
                             or "events" in full_path)
                if ((now_time - mtime) < STABLE_THRESHOLD) and not is_exempt:
                    continue
                rel_path = os.path.relpath(full_path, local_path)
                current_all_items.append(rel_path)
            except FileNotFoundError:
                continue

    # 3. 找出新增的文件
    new_items = [item for item in current_all_items if item not in uploaded_files]
    return new_items, uploaded_files


STEP_DIR_RE = re.compile(r"^global_step_(\d+)$")


def latest_complete_step(local_path, uploaded_set):
    """返回所有文件均已上传的最大 global_step N, 无则返回 None。"""
    candidates = []
    for name in os.listdir(local_path):
        m = STEP_DIR_RE.match(name)
        if m and os.path.isdir(os.path.join(local_path, name)):
            candidates.append(int(m.group(1)))

    for step in sorted(candidates, reverse=True):
        step_dir = os.path.join(local_path, f"global_step_{step}")
        complete = True
        for root, _, files in os.walk(step_dir):
            for fname in files:
                rel = os.path.relpath(os.path.join(root, fname), local_path)
                if rel not in uploaded_set:
                    complete = False
                    break
            if not complete:
                break
        if complete:
            return step
    return None


def push_tracker(remote_path, step):
    """幂等覆写远端 latest_checkpointed_iteration.txt 指向已完整上传的 step。"""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(str(step))
        tmp = f.name
    try:
        mox.file.copy(tmp, os.path.join(remote_path, "latest_checkpointed_iteration.txt"))
        print(f"[TRACKER] 远端追踪文件已更新 -> global_step_{step}")
    finally:
        os.unlink(tmp)


def loop_upload(local_path, remote_path, platform_url):
    record_file = os.path.join(local_path, "upload_record.txt")
    print(f"启动check-point增量上传: {local_path} -> {remote_path}")
    first_log_ok = False
    last_pushed_step = None

    if mox is None:
        print("[WARN] 当前环境没有 moxing, 上传进程空转 (仅本地调试)")

    while True:
        s_time = time.time()
        try:
            new_items, uploaded_files = get_files(record_file, local_path)
            success_items = []
            if new_items and mox is not None:
                print(f"[{datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')}] "
                      f"发现 {len(new_items)} 个新增项，开始上传...")
                for item in new_items:
                    src_item_path = os.path.join(local_path, item)
                    dst_item_path = os.path.join(remote_path, item)
                    try:
                        mox.file.copy_parallel(src_item_path, dst_item_path)

                        # 平台侧展示目录 (可选)
                        if platform_url:
                            platform_path = os.path.join(platform_url, item)
                            dest_dir = os.path.dirname(platform_path)
                            if not os.path.exists(dest_dir):
                                os.makedirs(dest_dir)
                            shutil.copy(src_item_path, platform_path)

                        # verl checkpoint 目录: global_step_N
                        if not first_log_ok and "global_step" in src_item_path:
                            PLAT_LOGGER.info("[Training Stage]output first loss done")
                            first_log_ok = True

                        exclude_cond = ("upload_record.txt" in item
                                        or "trainer_log.jsonl" in item or "logs" in item)
                        if exclude_cond or "events" in item:
                            continue
                        success_items.append(item)
                    except Exception as e:
                        print(f"上传出错 {item}: {e}")

                if success_items:
                    with open(record_file, "a", encoding="utf-8") as f:
                        f.write("\n".join(success_items) + "\n")
                    spend_time = time.time() - s_time
                    print(f"[{datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')}] "
                          f"已批量保存 {len(success_items)} 条上传记录。花费：{spend_time} s")
            else:
                print(f"[{datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')}] 未检测到变化。")

            # 每轮结束后: 若出现了新的"完整上传"的 step, 幂等覆写远端追踪文件
            if mox is not None:
                step = latest_complete_step(
                    local_path, uploaded_files | set(success_items))
                if step is not None and step != last_pushed_step:
                    try:
                        push_tracker(remote_path, step)
                        last_pushed_step = step
                    except Exception as e:
                        print(f"更新远端追踪文件出错: {e}")

        except Exception as e:
            print(f"监控进程发生异常: {e}")

        time.sleep(600)  # 等待 10 分钟


def main():
    parser = argparse.ArgumentParser(description="verl GRPO check-point 增量上传")
    parser.add_argument("--experiment_dir", type=str, required=True,
                        help="实验目录, 例如 experiments/qwendrive/collision_risk_grpo_01")
    args = parser.parse_args()

    exp_dir = os.path.abspath(args.experiment_dir)
    print("实验绝对路径：", exp_dir)

    config_path = os.path.join(exp_dir, "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    local_path = os.path.abspath(os.path.join(exp_dir, "outputs"))
    os.makedirs(local_path, exist_ok=True)
    remote_path = config["upload"]["check_point"]
    platform_url = os.environ.get("OUTPUT_URL")

    loop_upload(local_path, remote_path, platform_url)


if __name__ == "__main__":
    main()
