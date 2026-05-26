"""Copy a hard-example set's videos out for review, skipping already-reviewed ones.

hard_fn.json / hard_fp.json (from analyze_hard_examples.py) contain ALL errors
of that type, including the high-confidence ones you already sorted under
suspect_videos_after_check/. This script copies only the NOT-yet-reviewed
videos so you don't re-watch clips you already categorized.

Logic:
  1. Load --hard_json (list of training samples {messages, videos}).
  2. If --reviewed_dir is given, recursively collect every video stem under it
     (e.g. suspect_videos_after_check/**/*.mp4) → already-reviewed set.
  3. For each hard sample whose video stem is NOT already reviewed, copy the
     video into --dest_dir.
  4. Also write a JSON manifest (original training format) of the copied
     (still-to-review) samples, so they can be fed back into the pipeline later.

Usage:
    python copy_hard_for_review.py \\
        --hard_json    /.../corrections_or_hard_examples/hard_fn.json \\
        --reviewed_dir /.../hard_examples/suspect_videos_after_check \\
        --dest_dir     /.../hard_fn_to_review \\
        [--limit 0]    # 0 = no cap; >0 copies only the first N (after sort)

Matching is by video-path stem (basename without extension).
"""

import argparse
import json
import os
import shutil
from pathlib import Path

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def collect_reviewed_stems(reviewed_dir):
    """Recursively gather all video stems under reviewed_dir."""
    root = Path(reviewed_dir)
    stems = set()
    if not root.is_dir():
        print(f"⚠️  reviewed_dir not found: {reviewed_dir} (treating as none reviewed)")
        return stems
    for f in root.rglob("*"):
        if f.is_file() and f.suffix.lower() in VIDEO_EXTS:
            stems.add(f.stem)
    print(f"[reviewed] {reviewed_dir}")
    print(f"           {len(stems)} already-reviewed video stems collected")
    return stems


def load_hard(hard_json):
    with open(hard_json, "r", encoding="utf-8") as f:
        samples = json.load(f)
    print(f"[hard ] {hard_json}")
    print(f"        {len(samples)} samples")
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hard_json", required=True,
                    help="Hard-example JSON (e.g. hard_fn.json), list of {messages, videos}")
    ap.add_argument("--reviewed_dir", default=None,
                    help="Dir whose video stems are already reviewed (recursive). "
                         "Omit to copy ALL hard videos.")
    ap.add_argument("--dest_dir", required=True,
                    help="Where to copy the not-yet-reviewed videos (created if missing)")
    ap.add_argument("--manifest", default=None,
                    help="Optional path to write a JSON manifest of copied samples "
                         "(default: <dest_dir>/to_review.json)")
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, copy only the first N to-review videos (after input order)")
    args = ap.parse_args()

    samples = load_hard(args.hard_json)
    reviewed = collect_reviewed_stems(args.reviewed_dir) if args.reviewed_dir else set()

    os.makedirs(args.dest_dir, exist_ok=True)

    to_review = []          # samples not yet reviewed
    skipped_reviewed = 0
    no_video = 0
    for s in samples:
        videos = s.get("videos") or []
        if not videos:
            no_video += 1
            continue
        stem = Path(videos[0]).stem
        if stem in reviewed:
            skipped_reviewed += 1
            continue
        to_review.append((stem, videos[0], s))

    if args.limit and args.limit > 0:
        to_review = to_review[:args.limit]

    copied, missing, failed = 0, 0, 0
    for stem, src, _ in to_review:
        if not os.path.exists(src):
            missing += 1
            continue
        try:
            shutil.copy2(src, args.dest_dir)
            copied += 1
        except Exception as e:
            failed += 1
            print(f"  ❌ copy failed [{Path(src).name}]: {e}")

    manifest_path = args.manifest or os.path.join(args.dest_dir, "to_review.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump([s for _, _, s in to_review], f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("📦 Copy hard-examples for review")
    print("=" * 70)
    print(f"  total in hard_json     : {len(samples)}")
    print(f"  already reviewed (skip): {skipped_reviewed}")
    print(f"  no-video entries       : {no_video}")
    print(f"  to review (new)        : {len(to_review)}")
    print(f"    ↳ copied             : {copied}")
    print(f"    ↳ missing on disk    : {missing}")
    print(f"    ↳ copy failed        : {failed}")
    print(f"  dest_dir               : {args.dest_dir}")
    print(f"  manifest               : {manifest_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
