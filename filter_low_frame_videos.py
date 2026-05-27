"""Find videos with too few frames (frame-drop) → write a drop list (Path D, step 6).

Expected: 3s @ ~10fps ≈ 29–30 frames. A tiny fraction of clips have dropped
frames (e.g. 10–28). This scans the dataset, counts frames per video, prints
the distribution, and writes the stems of videos below --min_frames (plus any
unreadable ones) to a text file that build_final_dataset.py can consume via
--drop_stems_file.

Usage:
    python filter_low_frame_videos.py \\
        --dataset_json /.../train_all_label_corrected.json \\
        --min_frames 29 \\
        --out /.../low_frame_stems.txt

Notes:
  - Uses OpenCV's CAP_PROP_FRAME_COUNT (reads container metadata; fast but can
    be off by 1 on some codecs). For frame filtering at the 29/30 boundary that
    is fine. If you already have exact counts from another tool, you can skip
    this and write the drop list yourself (one stem or path per line).
"""

import argparse
import json
from collections import Counter
from pathlib import Path

try:
    import cv2
except ImportError:
    raise SystemExit("Need OpenCV: pip install opencv-python-headless")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_json", required=True,
                    help="Dataset JSON (list of {messages, videos})")
    ap.add_argument("--min_frames", type=int, default=29,
                    help="Videos with frame_count < this are dropped (default 29 → keeps 29/30)")
    ap.add_argument("--out", required=True, help="Output text file: one bad stem per line")
    args = ap.parse_args()

    with open(args.dataset_json, "r", encoding="utf-8") as f:
        samples = json.load(f)

    dist = Counter()
    bad, unreadable = [], []
    for i, s in enumerate(samples):
        videos = s.get("videos") or []
        if not videos:
            continue
        path = videos[0]
        stem = Path(path).stem
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            unreadable.append(stem)
            cap.release()
            continue
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        dist[n] += 1
        if n < args.min_frames:
            bad.append((stem, n))
        if (i + 1) % 5000 == 0:
            print(f"  ... scanned {i+1}/{len(samples)}")

    # distribution
    print("\n" + "=" * 50)
    print("帧数分布")
    print("=" * 50)
    print(f"{'frames':>8} | {'count':>8}")
    print("-" * 22)
    for n in sorted(dist):
        print(f"{n:>8} | {dist[n]:>8}")
    print(f"  unreadable: {len(unreadable)}")
    print("=" * 50)

    drop_stems = [s for s, _ in bad] + unreadable
    with open(args.out, "w", encoding="utf-8") as f:
        for s in drop_stems:
            f.write(s + "\n")

    print(f"\n  < {args.min_frames} frames : {len(bad)}")
    print(f"  unreadable     : {len(unreadable)}")
    print(f"  → wrote {len(drop_stems)} drop stems to {args.out}")


if __name__ == "__main__":
    main()
