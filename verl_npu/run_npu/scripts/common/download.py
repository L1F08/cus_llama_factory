# -*- coding: utf-8 -*-
"""OBS 下载工具: 与 llm_ft_longtime 的 config.yaml download 配置节完全兼容。

云端路径 (obs://...) 通过 moxing 下载 (仅 ModelArts 平台可用);
本地路径直接拷贝, 便于脱离平台调试。
"""

import os
import shutil

try:
    import moxing as mox  # ModelArts 平台内置
except ImportError:
    mox = None


def _copy(cloud_path, local_path):
    if cloud_path.startswith("obs://"):
        if mox is None:
            raise RuntimeError(
                f"需要下载 {cloud_path} 但当前环境没有 moxing (非 ModelArts 平台?)")
        mox.file.copy_parallel(cloud_path, local_path)
    elif os.path.isdir(cloud_path):
        shutil.copytree(cloud_path, local_path, dirs_exist_ok=True)
    else:
        shutil.copy(cloud_path, local_path)


def download_single(cloud_path, local_path, skip_if_exists=False):
    """单个路径下载 (支持文件和目录, 可选跳过已存在)。"""
    if skip_if_exists and os.path.exists(local_path):
        if os.path.isfile(local_path) or (os.path.isdir(local_path) and os.listdir(local_path)):
            print(f"[SKIP] 已存在，跳过下载: {local_path}")
            return

    print(f"[DOWNLOAD] {cloud_path} → {local_path}")
    os.makedirs(os.path.dirname(local_path) if not local_path.endswith("/") else local_path,
                exist_ok=True)
    _copy(cloud_path, local_path)


def download_all(download_cfg, data_type="train"):
    """统一下载模型、数据、check-point (与 llm_ft_longtime 同构)。"""
    if not download_cfg:
        return

    if download_cfg.get("model"):
        m = download_cfg["model"]
        download_single(m["cloud_path"], m["local_path"], m.get("skip_if_exists", False))
    print("模型下载完成！")

    if download_cfg.get("data"):
        for item in download_cfg["data"]:
            download_single(item["cloud_path"], item["local_path"],
                            item.get("skip_if_exists", False))
    print("数据下载完成！")

    if data_type == "test":
        if download_cfg.get("check-point"):
            for item in download_cfg["check-point"]:
                download_single(item["cloud_path"], item["local_path"],
                                item.get("skip_if_exists", False))
        print("推理check-point下载完成!")
    print("所有云端文件下载完成！")


def restore_checkpoint(remote_ckpt_dir, outputs_dir):
    """断点续训: 任务重启后从 OBS 恢复最近一个完整的 checkpoint。

    verl 的 trainer.resume_mode=auto 只认本地 outputs 下的
    latest_checkpointed_iteration.txt; ModelArts 重启后容器是全新的, 因此
    需要先把远端 (上传进程增量传上去的) checkpoint 拉回来。
    恢复是尽力而为: 任一步失败仅告警, 训练照常从头开始。
    """
    if mox is None:
        return
    tracker_local = os.path.join(outputs_dir, "latest_checkpointed_iteration.txt")
    if os.path.exists(tracker_local):
        print("[RESUME] 本地已有 checkpoint 追踪文件, 跳过远端恢复")
        return

    remote_ckpt_dir = remote_ckpt_dir.rstrip("/")
    tracker_remote = f"{remote_ckpt_dir}/latest_checkpointed_iteration.txt"
    try:
        if not mox.file.exists(tracker_remote):
            print("[RESUME] 远端无追踪文件, 从头训练")
            return

        # 远端追踪文件由上传进程保证只指向"完整上传"的 step; 仍按降序多留几个兜底
        tracker_step = int(mox.file.read(tracker_remote).strip())
        names = mox.file.list_directory(remote_ckpt_dir)
        steps = sorted(
            (int(n.split("global_step_")[1]) for n in names
             if n.startswith("global_step_") and n.split("global_step_")[1].isdigit()),
            reverse=True)
        candidates = [s for s in steps if s <= tracker_step] or steps

        for step in candidates:
            remote_step = f"{remote_ckpt_dir}/global_step_{step}"
            local_step = os.path.join(outputs_dir, f"global_step_{step}")
            print(f"[RESUME] 从远端恢复 checkpoint: {remote_step}")
            try:
                mox.file.copy_parallel(remote_step, local_step)
            except Exception as e:
                print(f"[RESUME] 恢复 global_step_{step} 失败: {e}, 尝试更早的 step")
                continue
            if not os.path.isdir(local_step) or not os.listdir(local_step):
                print(f"[RESUME] global_step_{step} 恢复后为空, 尝试更早的 step")
                continue

            with open(tracker_local, "w", encoding="utf-8") as f:
                f.write(str(step))

            # 把已恢复文件写入上传记录, 避免上传进程重复回传
            record_file = os.path.join(outputs_dir, "upload_record.txt")
            restored = []
            for root, _, files in os.walk(local_step):
                for fname in files:
                    restored.append(os.path.relpath(os.path.join(root, fname), outputs_dir))
            with open(record_file, "a", encoding="utf-8") as f:
                f.write("\n".join(restored) + "\n")

            print(f"[RESUME] 已恢复 global_step_{step} ({len(restored)} 个文件), "
                  f"verl 将自动续训")
            return
        print("[RESUME] 远端无可用完整 checkpoint, 从头训练")
    except Exception as e:
        print(f"[RESUME] 远端恢复失败 (将从头训练): {e}")
