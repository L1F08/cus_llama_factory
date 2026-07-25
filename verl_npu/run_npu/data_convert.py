#!/usr/bin/python3.10
# -*- coding: utf-8 -*-
"""LlamaFactory(sharegpt) 训练 JSON -> verl RLHFDataset parquet 转换器。

输入格式 (与 llm_ft_longtime / data_process/video_gen/dataset_format.py 产出一致):
    [
      {
        "messages": [
          {"role": "user", "content": "<video>\n...任务提示词..."},
          {"role": "assistant", "content": "高风险"}          # 或 "安全"
        ],
        "videos": ["/path/to/cam_front.mp4", ...]
      },
      ...
    ]

同时兼容测试集的 content-block 格式 (无 assistant 答案的样本会被跳过并计数):
    [
      [{"role": "user", "content": [{"type": "video", "video": "...", ...},
                                    {"type": "text", "text": "..."}]}],
      ...
    ]

输出 parquet 行结构 (对应 verl/utils/dataset/rl_dataset.py::RLHFDataset):
    data_source   : str  — 自定义 reward 场景下仅作标识
    prompt        : [{"role": "user", "content": "<video>\n...提示词"}]
    videos        : [{"video": 绝对路径, "fps": F, "max_frames": N, "max_pixels": P}]
    ability       : str
    reward_model  : {"style": "rule", "ground_truth": "高风险"/"安全"}
    extra_info    : {"split": ..., "index": ..., "videos": [...]}

其中 videos 条目的 fps / max_frames / max_pixels 会被 verl 原样透传给
qwen_vl_utils.process_vision_info, 控制视频抽帧与分辨率。
"""

import argparse
import json
import os
import re
import random

VALID_LABELS = ("高风险", "安全")
# 非视频占位符必须清除: verl RLHFDataset 会按 <image>/<audio> 数量断言对应列长度,
# 本任务只有 videos 列, 残留的 <image>/<audio> 会让样本在训练侧被静默过滤
NON_VIDEO_PLACEHOLDER_RE = re.compile(r"<image>|<audio>")


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _remap_video_path(path, video_root=None, path_map=None):
    """视频路径重映射: 先做前缀替换, 再对相对路径拼接 video_root, 最终转绝对路径。"""
    if path_map:
        for src, dst in path_map:
            if path.startswith(src):
                path = dst + path[len(src):]
                break
    if video_root and not os.path.isabs(path):
        path = os.path.join(video_root, path)
    return os.path.abspath(path)


def _build_video_entry(path, fps=None, max_frames=None, max_pixels=None):
    entry = {"video": path}
    if fps is not None:
        entry["fps"] = float(fps)
    if max_frames is not None:
        entry["max_frames"] = int(max_frames)
    if max_pixels is not None:
        entry["max_pixels"] = int(max_pixels)
    return entry


def _parse_sample(item):
    """解析单条样本, 统一返回 (user_text, label, video_paths) 三元组。

    label 为 None 表示该样本没有可用的 ground truth (如 content-block 测试集)。
    """
    # 格式 A: {"messages": [...], "videos": [...]}  (训练集标准格式)
    if isinstance(item, dict) and "messages" in item:
        user_text, label = None, None
        for msg in item["messages"]:
            if msg["role"] == "user" and user_text is None:
                user_text = msg["content"]
            elif msg["role"] == "assistant" and label is None:
                label = str(msg["content"]).strip()
        videos = list(item.get("videos", []))
        return user_text, label, videos

    # 格式 B: [{"role": "user", "content": [blocks]}]  (content-block 测试集)
    if isinstance(item, list):
        user_text_parts, videos = [], []
        for msg in item:
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, str):
                user_text_parts.append(content)
                continue
            for block in content:
                if block.get("type") == "video":
                    videos.append(block["video"])
                    user_text_parts.append("<video>\n")
                elif block.get("type") == "text":
                    user_text_parts.append(block["text"])
        return "".join(user_text_parts), None, videos

    raise ValueError(f"无法识别的样本格式: {type(item)}")


def _normalize_placeholders(user_text, n_videos):
    """确保 user_text 中占位符与媒体列严格对应。

    verl 的 RLHFDataset 要求每类占位符个数与对应媒体列长度严格相等:
    - 先清除 <image>/<audio> (本任务没有对应列, 残留会导致样本被静默过滤);
    - <video> 数量不一致时: 清除原占位符, 统一在开头补 n 个 "<video>\n"。
    """
    user_text, n_removed = NON_VIDEO_PLACEHOLDER_RE.subn("", user_text)
    count = len(re.findall(r"<video>", user_text))
    if count == n_videos:
        return user_text, n_removed > 0
    cleaned = re.sub(r"<video>\n?", "", user_text)
    return "<video>\n" * n_videos + cleaned, True


def convert_samples(
    raw,
    split,
    data_source="collision_risk",
    ability="video_risk_cls",
    video_root=None,
    path_map=None,
    fps=None,
    max_frames=None,
    max_pixels=None,
    check_video_exists=True,
):
    """核心转换。返回 (rows, stats)。"""
    rows = []
    stats = {"total": len(raw), "kept": 0, "no_label": 0, "bad_label": 0,
             "fixed_placeholder": 0, "missing_video": 0,
             "first_missing_video": None}

    for idx, item in enumerate(raw):
        user_text, label, videos = _parse_sample(item)
        if not user_text or not videos:
            stats["no_label"] += 1
            continue
        if label is None:
            stats["no_label"] += 1
            continue
        if label not in VALID_LABELS:
            stats["bad_label"] += 1
            print(f"[WARN] 样本 {idx} 答案非法(既非高风险也非安全): {label!r}, 已跳过")
            continue

        video_paths = [_remap_video_path(v, video_root, path_map) for v in videos]
        if check_video_exists:
            missing = [v for v in video_paths if not os.path.exists(v)]
            if missing:
                stats["missing_video"] += 1
                if stats["first_missing_video"] is None:
                    stats["first_missing_video"] = missing[0]
                print(f"[WARN] 样本 {idx} 视频缺失: {missing[0]}, 已跳过")
                continue

        user_text, fixed = _normalize_placeholders(user_text, len(video_paths))
        if fixed:
            stats["fixed_placeholder"] += 1

        rows.append({
            "data_source": data_source,
            "prompt": [{"role": "user", "content": user_text}],
            "videos": [_build_video_entry(v, fps, max_frames, max_pixels)
                       for v in video_paths],
            "ability": ability,
            "reward_model": {"style": "rule", "ground_truth": label},
            "extra_info": {
                "split": split,
                "index": idx,
                "videos": video_paths,
            },
        })
        stats["kept"] += 1

    return rows, stats


def assert_no_missing_videos(stats, where):
    """视频缺失通常是 video_root/path_map 配置错误, 会静默缩小数据集 — 直接失败。"""
    if stats.get("missing_video"):
        raise RuntimeError(
            f"{where}: {stats['missing_video']} 条样本的视频文件缺失 "
            f"(首个: {stats.get('first_missing_video')})。"
            f"请检查 convert.video_root / convert.path_map 配置; "
            f"若确认允许缺失, 可设置 check_video_exists: false")


def save_parquet(rows, out_path):
    """优先用 datasets(与 verl 官方预处理脚本一致), 缺失时回退 pandas。"""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    try:
        from datasets import Dataset
        Dataset.from_list(rows).to_parquet(out_path)
    except ImportError:
        import pandas as pd
        pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"[OK] 已写出 {len(rows)} 条 -> {out_path}")


def convert_file(
    in_json,
    out_parquet,
    split,
    **kwargs,
):
    raw = _load_json(in_json)
    rows, stats = convert_samples(raw, split, **kwargs)
    print(f"[STATS] {split}: {stats}")
    if not rows:
        return []
    save_parquet(rows, out_parquet)
    return rows


def split_train_val(rows, val_size, seed=42):
    """从训练行中切一小部分作验证 (测试集无标签时的兜底)。"""
    n_val = max(1, int(len(rows) * val_size)) if val_size < 1 else int(val_size)
    n_val = min(n_val, len(rows) - 1)
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    val_idx = set(indices[:n_val])
    train_rows = [r for i, r in enumerate(rows) if i not in val_idx]
    val_rows = [r for i, r in enumerate(rows) if i in val_idx]
    for r in val_rows:
        r["extra_info"]["split"] = "val"
    return train_rows, val_rows


def _parse_path_map(items):
    pairs = []
    for it in items or []:
        if ":" not in it:
            raise ValueError(f"--path_map 需为 src_prefix:dst_prefix 形式, 得到: {it}")
        src, dst = it.split(":", 1)
        pairs.append((src, dst))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="LlamaFactory JSON -> verl parquet")
    parser.add_argument("--train_json", type=str, required=True)
    parser.add_argument("--test_json", type=str, default=None,
                        help="可选; 若无标签或缺失则自动从训练集切 val")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--data_source", type=str, default="collision_risk")
    parser.add_argument("--ability", type=str, default="video_risk_cls")
    parser.add_argument("--video_root", type=str, default=None,
                        help="相对视频路径的根目录")
    parser.add_argument("--path_map", type=str, nargs="*", default=None,
                        help="路径前缀替换, 形式 src_prefix:dst_prefix, 可多个")
    parser.add_argument("--fps", type=float, default=8.0,
                        help="抽帧率, 透传给 qwen_vl_utils (与 SFT video_fps 对齐)")
    parser.add_argument("--max_frames", type=int, default=24,
                        help="3s x 8fps = 24 帧")
    parser.add_argument("--max_pixels", type=int, default=589824,
                        help="单帧最大像素 (768², 与 SFT video_max_pixels 对齐)")
    parser.add_argument("--val_size", type=float, default=0.05,
                        help="测试集不可用时, 从训练集切出的验证比例(或条数)")
    parser.add_argument("--check_video_exists", action=argparse.BooleanOptionalAction,
                        default=True, help="转换时校验视频存在, 缺失则报错")
    args = parser.parse_args()

    common = dict(
        data_source=args.data_source,
        ability=args.ability,
        video_root=args.video_root,
        path_map=_parse_path_map(args.path_map),
        fps=args.fps,
        max_frames=args.max_frames,
        max_pixels=args.max_pixels,
        check_video_exists=args.check_video_exists,
    )

    train_rows, train_stats = convert_samples(
        _load_json(args.train_json), "train", **common)
    print(f"[STATS] train: {train_stats}")
    assert_no_missing_videos(train_stats, "train")
    if not train_rows:
        raise RuntimeError("训练集转换后为空, 请检查输入 JSON 格式")

    val_rows = []
    if args.test_json and os.path.exists(args.test_json):
        val_rows, val_stats = convert_samples(
            _load_json(args.test_json), "val", **common)
        print(f"[STATS] val: {val_stats}")
        assert_no_missing_videos(val_stats, "val")

    if not val_rows:
        print(f"[INFO] 测试集不可用(缺失或无标签), 从训练集切 {args.val_size} 作验证")
        train_rows, val_rows = split_train_val(train_rows, args.val_size)

    train_path = os.path.join(args.out_dir, "train.parquet")
    val_path = os.path.join(args.out_dir, "val.parquet")
    save_parquet(train_rows, train_path)
    save_parquet(val_rows, val_path)
    print(f"[DONE] train={len(train_rows)} val={len(val_rows)}")
    return train_path, val_path


if __name__ == "__main__":
    main()
