# -*- coding: utf-8 -*-
"""OBS 上传工具: 训练结束后整体上传实验产物 (与 llm_ft_longtime 同构)。"""

import os
from datetime import datetime
from zoneinfo import ZoneInfo

try:
    import moxing as mox
except ImportError:
    mox = None


def upload_experiment_results(exp_dir, upload_cfg):
    """上传 outputs/ 与 logs/ 到云端。"""
    if not upload_cfg or not upload_cfg.get("base_obs_path"):
        print("未配置上传路径，跳过上传")
        return

    if mox is None:
        print("当前环境没有 moxing, 跳过上传")
        return

    base_obs = upload_cfg["base_obs_path"].rstrip("/")
    exp_name = os.path.basename(exp_dir)
    timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    target_obs = f"{base_obs}/{exp_name}/{timestamp}/"

    print(f"[UPLOAD] 上传实验结果到 {target_obs}")
    if os.path.exists(exp_dir) and os.listdir(exp_dir):
        mox.file.copy_parallel(exp_dir, f"{target_obs}/")
    print(f"上传完成！云端路径: {target_obs}")
