"""Check per-sample multi-cam frame consistency in a training JSON.

For each sample (with a `videos` list of N camera paths), decode every video and
verify:
  1. all N cameras have the SAME frame count (misalignment → Qwen "video features
     and video tokens do not match" at train time);
  2. the frame count is EVEN (temporal_patch_size=2 wants pairs; odd frames are a
     known source of the same token/feature mismatch);
  3. no unreadable / missing video.

Uses a DECODE count (cv2 grab loop) by default — container metadata
(CAP_PROP_FRAME_COUNT) over-reports and would hide real drops. Multiprocessing.

Outputs:
  - console summary (how many samples are inconsistent / odd / unreadable)
  - --out_csv : per-sample record (id, per-cam frames, consistent, even, status)
  - --bad_stems : newline list of bad sample stems (feed to build_final_dataset
                  --drop_stems_file to exclude them)

Usage:
    python check_multicam_frames.py \\
        --dataset_json /path/to/train.json \\
        --workers 180 \\
        --out_csv /path/to/multicam_frames.csv \\
        --bad_stems /path/to/multicam_bad_stems.txt
"""

import argparse
import csv
import json
import os
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

try:
    import cv2
except ImportError:
    raise SystemExit("Need OpenCV: pip install opencv-python-headless")


def _init_worker(method):
    global _METHOD
    _METHOD = method
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass


def _count(path):
    """Frame count. -1 if unreadable/missing."""
    if not path or not os.path.exists(path):
        return -1
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return -1
    if _METHOD == "meta":
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return n
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n


def _worker(item):
    """item = (stem, [paths]). Returns (stem, [frames], [paths])."""
    stem, paths = item
    return stem, [_count(p) for p in paths], paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_json", required=True, help="Training JSON (list of {messages, videos})")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--method", choices=["decode", "meta"], default="decode",
                    help="decode = real grab loop (reliable); meta = container count (fast, unreliable)")
    ap.add_argument("--require_even", action="store_true", default=True,
                    help="flag odd frame counts as bad (temporal_patch_size=2). Default on.")
    ap.add_argument("--allow_odd", dest="require_even", action="store_false",
                    help="do NOT flag odd frame counts")
    ap.add_argument("--out_csv", default=None, help="per-sample CSV (default: <dataset_dir>/multicam_frames.csv)")
    ap.add_argument("--bad_stems", default=None, help="output: newline list of bad sample stems")
    args = ap.parse_args()

    with open(args.dataset_json, "r", encoding="utf-8") as f:
        samples = json.load(f)

    items, no_video = [], 0
    for s in samples:
        videos = s.get("videos") or []
        if not videos:
            no_video += 1
            continue
        stem = Path(videos[0]).stem
        items.append((stem, list(videos)))

    ncam = Counter(len(p) for _, p in items)
    print(f"[load] {len(items)} samples (skipped {no_video} without video); "
          f"cam-count distribution: {dict(ncam)}")
    print(f"       method={args.method}, workers={args.workers}, require_even={args.require_even}")

    results = []
    with Pool(args.workers, initializer=_init_worker, initargs=(args.method,)) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker, items, chunksize=16)):
            results.append(r)
            if (i + 1) % 5000 == 0:
                print(f"  ... {i + 1}/{len(items)}")

    # classify
    bad_stems, n_unreadable, n_inconsistent, n_odd, n_ok = [], 0, 0, 0, 0
    frame_dist = Counter()
    rows = []
    for stem, frames, paths in results:
        status = []
        if any(fr == -1 for fr in frames):
            status.append("unreadable")
        good = [fr for fr in frames if fr != -1]
        consistent = len(set(good)) <= 1 and len(good) == len(frames)
        if not consistent and "unreadable" not in status:
            status.append("inconsistent")
        even = all(fr % 2 == 0 for fr in good) if good else False
        if args.require_even and good and not even:
            status.append("odd")
        if good and consistent:
            frame_dist[good[0]] += 1

        is_bad = len(status) > 0
        if "unreadable" in status:
            n_unreadable += 1
        if "inconsistent" in status:
            n_inconsistent += 1
        if "odd" in status:
            n_odd += 1
        if not is_bad:
            n_ok += 1
        else:
            bad_stems.append(stem)
        rows.append((stem, frames, int(consistent), int(even), ";".join(status) or "ok"))

    # report
    print("\n" + "=" * 66)
    print("📊 多相机帧数一致性检查")
    print("=" * 66)
    print(f"  样本总数            : {len(results)}")
    print(f"  ✅ 全部正常          : {n_ok}")
    print(f"  ❌ 帧数不一致        : {n_inconsistent}")
    print(f"  ❌ 帧数为奇数        : {n_odd}" + ("" if args.require_even else "（未检查）"))
    print(f"  ❌ 有视频读不出/缺失 : {n_unreadable}")
    print(f"  → 需剔除(bad)总数    : {len(bad_stems)}")
    if frame_dist:
        print("\n  --- 一致样本的帧数分布 ---")
        for fr in sorted(frame_dist):
            tag = "  ← 奇数" if fr % 2 else ""
            print(f"    {fr:>4} 帧: {frame_dist[fr]:>7}{tag}")
    print("=" * 66)

    # sample a few bad ones for eyeballing
    bad_rows = [r for r in rows if r[4] != "ok"]
    if bad_rows:
        print("  前几个异常样本:")
        for stem, frames, _, _, st in bad_rows[:8]:
            print(f"    {stem}  frames={frames}  [{st}]")

    out_csv = args.out_csv or str(Path(args.dataset_json).with_name("multicam_frames.csv"))
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stem", "frames", "consistent", "even", "status"])
        for stem, frames, cons, even, st in rows:
            w.writerow([stem, "|".join(map(str, frames)), cons, even, st])
    print(f"\n  ↳ 每样本记录: {out_csv}")

    if args.bad_stems:
        with open(args.bad_stems, "w", encoding="utf-8") as f:
            for s in bad_stems:
                f.write(s + "\n")
        print(f"  ↳ 异常 stem 清单: {args.bad_stems}（可作 build_final_dataset 的 --drop_stems_file）")


if __name__ == "__main__":
    main()
