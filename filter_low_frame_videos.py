"""Count frames per video (multiprocessing) → distribution + per-video CSV + drop list.

Path D, step 6.

IMPORTANT: defaults to a DECODE count (cv2 grab loop), NOT container metadata.
cv2's CAP_PROP_FRAME_COUNT reads the container's nominal frame count, which is
frequently WRONG — many real frame-dropped clips still report 29/30 in metadata
and therefore slip through a metadata-based filter. The decode count walks the
actual decodable frames, so it catches them. Use --method meta for the fast
(unreliable) metadata count if you really want it.

Outputs:
  --counts_csv : every video's stem,path,frames  (full record)
  --out        : drop list — stems with frames <= --drop_max_frames (default 28)
                 PLUS unreadable videos (frames == -1). Consumed by
                 build_final_dataset.py via --drop_stems_file.

Usage:
    python filter_low_frame_videos.py \\
        --dataset_json /.../train_all_label_corrected.json \\
        --drop_max_frames 28 \\
        --workers 192 \\
        --counts_csv /.../frame_counts.csv \\
        --out        /.../low_frame_stems.txt
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


# ---- frame counting (runs in worker processes) ----
_METHOD = "decode"


def _init_worker(method):
    global _METHOD
    _METHOD = method
    # keep each worker single-threaded so 192 procs don't oversubscribe
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass


def _count_decode(path):
    """Actual decodable frame count via grab loop. -1 if unreadable."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return -1
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n


def _count_meta(path):
    """Container-metadata frame count (fast, unreliable). -1 if unreadable."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return -1
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def _worker(item):
    stem, path = item
    fn = _count_decode if _METHOD == "decode" else _count_meta
    return stem, path, fn(path)


# ---- pure aggregation (no cv2; unit-testable) ----
def summarize_and_write(results, counts_csv, out_path, drop_max_frames):
    """results: list of (stem, path, frames). Writes per-video CSV + drop list,
    prints distribution. Returns (n_drop_lowframe, n_unreadable)."""
    # per-video record
    with open(counts_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stem", "path", "frames"])
        for stem, path, n in sorted(results, key=lambda r: r[2]):
            w.writerow([stem, path, n])

    dist = Counter(n for _, _, n in results)
    drop_stems, unreadable = [], 0
    for stem, _, n in results:
        if n == -1:
            drop_stems.append(stem)
            unreadable += 1
        elif n <= drop_max_frames:
            drop_stems.append(stem)

    with open(out_path, "w", encoding="utf-8") as f:
        for s in drop_stems:
            f.write(s + "\n")

    # distribution print
    print("\n" + "=" * 40)
    print("帧数分布 (frame distribution)")
    print("=" * 40)
    print(f"{'frames':>8} | {'count':>8}")
    print("-" * 22)
    for n in sorted(dist):
        tag = "  ← unreadable" if n == -1 else ("  ← DROP" if n <= drop_max_frames else "")
        print(f"{n:>8} | {dist[n]:>8}{tag}")
    print("=" * 40)
    n_low = len(drop_stems) - unreadable
    print(f"  <= {drop_max_frames} frames : {n_low}")
    print(f"  unreadable      : {unreadable}")
    print(f"  → drop total    : {len(drop_stems)}")
    print(f"  ↳ per-video counts : {counts_csv}")
    print(f"  ↳ drop list        : {out_path}")
    return n_low, unreadable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_json", required=True,
                    help="Dataset JSON (list of {messages, videos})")
    ap.add_argument("--drop_max_frames", type=int, default=28,
                    help="Drop videos with frames <= this (default 28 → keeps >=29)")
    ap.add_argument("--workers", type=int, default=os.cpu_count(),
                    help=f"Process count (default = all CPUs = {os.cpu_count()})")
    ap.add_argument("--method", choices=["decode", "meta"], default="decode",
                    help="decode = actual frame walk (reliable); meta = container count (fast, unreliable)")
    ap.add_argument("--counts_csv", default=None,
                    help="Per-video frame-count CSV (default: <dataset_dir>/frame_counts.csv)")
    ap.add_argument("--out", required=True, help="Drop-list output (one bad stem per line)")
    args = ap.parse_args()

    with open(args.dataset_json, "r", encoding="utf-8") as f:
        samples = json.load(f)

    # Multi-cam aware: scan EVERY video of each sample, all keyed by the sample's
    # shared stem (front/left/right share one autoscene id, different dirs). The
    # drop list drops the stem if ANY camera is low-frame/unreadable (a sample
    # missing a view can't be trained).
    items, no_video = [], 0
    for s in samples:
        videos = s.get("videos") or []
        if not videos:
            no_video += 1
            continue
        stem = Path(videos[0]).stem  # sample id, shared across cameras
        for v in videos:
            items.append((stem, v))
    n_samples = len(samples) - no_video
    print(f"[load] {n_samples} samples → {len(items)} video files to scan "
          f"(avg {len(items)/max(n_samples,1):.1f} cam/sample, skipped {no_video} without video), "
          f"method={args.method}, workers={args.workers}")

    results = []
    with Pool(processes=args.workers, initializer=_init_worker, initargs=(args.method,)) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker, items, chunksize=16)):
            results.append(r)
            if (i + 1) % 5000 == 0:
                print(f"  ... {i + 1}/{len(items)}")

    counts_csv = args.counts_csv or str(Path(args.dataset_json).with_name("frame_counts.csv"))
    summarize_and_write(results, counts_csv, args.out, args.drop_max_frames)


if __name__ == "__main__":
    main()
