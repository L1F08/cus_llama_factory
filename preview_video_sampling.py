"""Preview what the model ACTUALLY sees per (fps, resolution) — for eyeballing
whether a setting captures a fast (0.5-1s) collision-risk event.

Replicates LlamaFactory / Qwen video preprocessing:
  1. frame sampling: sample_frames = floor(duration * target_fps),
     picked via linspace(0, total_frames-1, sample_frames)   [mm_plugin.py]
  2. resolution: Qwen smart_resize to video_max_pixels (dims rounded to
     factor = patch_size(16) * spatial_merge_size(2) = 32; area <= max_pixels)

For each video × fps × max_pixels it writes a contact sheet PNG: the sampled
frames in order, resized to the REAL model resolution, each labeled with its
timestamp. Also prints frame count, inter-frame spacing, and token budget.

Usage:
    # from a data json (one sample) — uses that sample's `videos` list
    python preview_video_sampling.py \\
        --sample_json /path/train.json --index 0 \\
        --fps 3 4 5 6 --max_pixels 401408 589824 \\
        --out_dir /path/preview_out

    # or pass video paths directly
    python preview_video_sampling.py \\
        --videos front.mp4 left.mp4 right.mp4 \\
        --fps 3 5 --max_pixels 401408 --out_dir /path/preview_out
"""

import argparse
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np

PATCH_SIZE = 16
SPATIAL_MERGE = 2
FACTOR = PATCH_SIZE * SPATIAL_MERGE            # 32 for Qwen3.5-VL
TEMPORAL_PATCH = 2


def smart_resize(h, w, factor=FACTOR, max_pixels=401408, min_pixels=FACTOR * FACTOR):
    """Qwen smart_resize: keep aspect ratio, area <= max_pixels, dims multiple of factor."""
    h_bar = max(factor, round(h / factor) * factor)
    w_bar = max(factor, round(w / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((h * w) / max_pixels)
        h_bar = max(factor, math.floor(h / beta / factor) * factor)
        w_bar = max(factor, math.floor(w / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        h_bar = math.ceil(h * beta / factor) * factor
        w_bar = math.ceil(w * beta / factor) * factor
    return int(h_bar), int(w_bar)


def probe(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    duration = (total / src_fps) if src_fps > 0 else 0.0
    return dict(path=path, total=total, src_fps=src_fps, duration=duration, h=h, w=w)


def sample_indices(total, duration, target_fps):
    """Replicate mm_plugin: floor(duration*fps) frames via linspace over total."""
    n = max(1, math.floor(duration * target_fps))
    n = min(total, n)
    return np.linspace(0, total - 1, n).astype(np.int32)


def read_frames(path, indices):
    cap = cv2.VideoCapture(path)
    frames = {}
    want = set(int(i) for i in indices)
    i = 0
    # sequential grab is more reliable than random seek on some codecs
    ok = True
    while ok and want:
        ok, frame = cap.read()
        if not ok:
            break
        if i in want:
            frames[i] = frame
            want.discard(i)
        i += 1
    cap.release()
    return [frames.get(int(idx)) for idx in indices]


def contact_sheet(frames, indices, src_fps, out_hw, per_row, label_h=26):
    """Grid of frames resized to out_hw, each with a timestamp label bar."""
    H, W = out_hw
    cells = []
    for frame, idx in zip(frames, indices):
        if frame is None:
            cell = np.zeros((H + label_h, W, 3), np.uint8)
            cv2.putText(cell, "MISSING", (4, H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cells.append(cell)
            continue
        r = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        bar = np.full((label_h, W, 3), 40, np.uint8)
        t = (idx / src_fps) if src_fps > 0 else 0.0
        cv2.putText(bar, f"t={t:.2f}s f{int(idx)}", (4, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cells.append(np.vstack([bar, r]))

    rows = []
    for i in range(0, len(cells), per_row):
        chunk = cells[i:i + per_row]
        while len(chunk) < per_row:
            chunk.append(np.zeros_like(cells[0]))
        rows.append(np.hstack(chunk))
    return np.vstack(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", help="视频路径（直接给三路）")
    ap.add_argument("--sample_json", help="数据 json（取其中一条的 videos）")
    ap.add_argument("--index", type=int, default=0, help="取 json 第几条（默认 0）")
    ap.add_argument("--fps", type=float, nargs="+", required=True, help="要试的 fps 列表")
    ap.add_argument("--max_pixels", type=int, nargs="+", required=True, help="要试的 video_max_pixels 列表")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--per_row", type=int, default=6, help="接触表每行帧数")
    args = ap.parse_args()

    if args.videos:
        videos = args.videos
    elif args.sample_json:
        data = json.load(open(args.sample_json, encoding="utf-8"))
        videos = data[args.index].get("videos") or []
        print(f"[sample] json[{args.index}] videos: {videos}")
    else:
        raise SystemExit("❌ 需要 --videos 或 --sample_json")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for path in videos:
        info = probe(path)
        cam = Path(path).stem[:8]
        if info is None:
            print(f"❌ 打不开: {path}")
            continue
        print("\n" + "=" * 74)
        print(f"📹 {path}")
        print(f"   源: {info['total']} 帧 @ {info['src_fps']:.2f}fps = {info['duration']:.2f}s, "
              f"{info['w']}x{info['h']}")
        print("=" * 74)
        print(f"   {'fps':>4} | {'抽帧数':>6} | {'相邻间隔':>8} | {'末段1s帧数':>9} | "
              f"{'分辨率(HxW)':>12} | {'patch':>9} | {'tok/帧':>6} | {'总tok':>7}")
        print("   " + "-" * 82)

        for fps in args.fps:
            idx = sample_indices(info["total"], info["duration"], fps)
            n = len(idx)
            spacing = info["duration"] / n if n else 0
            # 末段最后 1 秒里有几帧被采到
            last1s = sum(1 for i in idx if (i / info["src_fps"]) >= info["duration"] - 1.0) if info["src_fps"] else 0
            for mp in args.max_pixels:
                Hb, Wb = smart_resize(info["h"], info["w"], max_pixels=mp)
                patches = (Hb // PATCH_SIZE) * (Wb // PATCH_SIZE)
                tok_per_frame = patches // (SPATIAL_MERGE ** 2)
                tok_total = tok_per_frame * n  # 近似（temporal 合并会再约减半，此为上界）
                print(f"   {fps:>4.1f} | {n:>6} | {spacing:>7.2f}s | {last1s:>9} | "
                      f"{Hb:>5}x{Wb:<6} | {patches:>9} | {tok_per_frame:>6} | {tok_total:>7}")

                frames = read_frames(path, idx)
                sheet = contact_sheet(frames, idx, info["src_fps"], (Hb, Wb), args.per_row)
                fn = out / f"{cam}_fps{fps:g}_px{mp}.png"
                cv2.imwrite(str(fn), sheet)

        print(f"   ↳ 接触表 PNG 已写入 {out}/{cam}_fps*_px*.png")

    print("\n人工核对要点：")
    print("  · '相邻间隔' > 0.5s 时，0.5s 内发生的风险可能落在两帧之间 → fps 偏低")
    print("  · '末段1s帧数' 太少 → 触发前最后1秒（最关键）被欠采样")
    print("  · 看 PNG 里目标（前车/侧方VRU）在该分辨率下是否看得清")


if __name__ == "__main__":
    main()
