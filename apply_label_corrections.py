"""Apply manual-review results back to the full dataset (Path D, step 2).

After eyeballing the suspect-label-noise clips you sorted them into:

    suspect_videos_after_check/
    ├── label1_risk/   (orig label = 1 高风险, model said 安全)
    │   ├── true_risk/      标签对, 真危险 → model wrong, genuine hard FN (keep label 1)
    │   ├── fake_risk/      标签错, 实际安全 → FLIP 1 → 0
    │   ├── not_sure/       排除
    │   └── low_quality/    排除
    └── label0_safe/   (orig label = 0 安全, model said 高风险)
        ├── true_safe/      标签对, 真安全 → model wrong, genuine hard FP (keep label 0)
        ├── fake_safe/      标签错, 实际危险 → FLIP 0 → 1
        ├── not_sure/       排除
        └── low_quality/    排除

This script:
  1. Scans those folders (by video-file stem) to build a correction table.
  2. Writes the fake-label corrections (orig vs corrected) to CSV + JSON for the record.
  3. Produces a label-corrected copy of the FULL dataset json (fakes flipped).
  4. Compiles the genuine hard examples (true_risk / true_safe) into JSON,
     ready for a later oversampling decision (factor not fixed here).
  5. Compiles the exclude list (not_sure + low_quality) — NOT removed from the
     corrected json unless --drop_exclude is given.

Label convention: assistant content "0" = 安全, "1" = 高风险.
Matching between folders and dataset is by video-path stem (basename w/o ext).

Usage:
    python apply_label_corrections.py \\
        --review_dir /home/ma-user/work/lyf/.../suspect_videos_after_check \\
        --full_json  /home/ma-user/work/lyf/.../train_all_..._cleaned.json \\
        --out_dir    /home/ma-user/work/lyf/.../corrections \\
        [--drop_exclude]
"""

import argparse
import copy
import csv
import json
import os
from pathlib import Path

LABEL_TEXT = {0: "安全", 1: "高风险"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# (parent_folder, sub_folder) -> (kind, orig_label, corrected_label)
#   kind: "true"=keep, "fake"=flip, "exclude"=drop-candidate
CATEGORY_MAP = {
    ("label1_risk", "true_risk"):   ("true",    1, 1),
    ("label1_risk", "fake_risk"):   ("fake",    1, 0),
    ("label1_risk", "not_sure"):    ("exclude", 1, None),
    ("label1_risk", "low_quality"): ("exclude", 1, None),
    ("label0_safe", "true_safe"):   ("true",    0, 0),
    ("label0_safe", "fake_safe"):   ("fake",    0, 1),
    ("label0_safe", "not_sure"):    ("exclude", 0, None),
    ("label0_safe", "low_quality"): ("exclude", 0, None),
}


def get_assistant_and_label(sample):
    """Return (assistant_msg_dict, int_label)."""
    assistant = next(m for m in sample["messages"] if m.get("role") == "assistant")
    raw = str(assistant["content"]).strip()
    if raw in ("0", "安全"):
        return assistant, 0
    if raw in ("1", "高风险"):
        return assistant, 1
    return assistant, int(raw)


def scan_review_dir(review_dir):
    """Return list of dicts: stem -> {parent, sub, kind, orig, corrected, video_file}."""
    review = Path(review_dir)
    entries = {}
    dupes = []
    for (parent, sub), (kind, orig, corrected) in CATEGORY_MAP.items():
        d = review / parent / sub
        if not d.is_dir():
            continue
        for f in d.iterdir():
            if f.suffix.lower() not in VIDEO_EXTS:
                continue
            stem = f.stem
            if stem in entries:
                dupes.append((stem, entries[stem]["parent_sub"], f"{parent}/{sub}"))
                continue
            entries[stem] = {
                "stem": stem,
                "parent_sub": f"{parent}/{sub}",
                "kind": kind,
                "orig": orig,
                "corrected": corrected,
                "video_file": str(f),
            }
    if dupes:
        print(f"⚠️  {len(dupes)} stems appear in multiple folders (kept first):")
        for s, a, b in dupes[:10]:
            print(f"    {s}: {a} vs {b}")
    return entries


def load_full(full_json):
    with open(full_json, "r", encoding="utf-8") as f:
        samples = json.load(f)
    stem_to_idx = {}
    bad = 0
    for i, s in enumerate(samples):
        videos = s.get("videos") or []
        if not videos:
            bad += 1
            continue
        stem_to_idx[Path(videos[0]).stem] = i
    print(f"[full ] {full_json}")
    print(f"        {len(samples)} samples ({len(stem_to_idx)} with a video, {bad} without)")
    return samples, stem_to_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--review_dir", required=True,
                    help="suspect_videos_after_check directory with the sorted subfolders")
    ap.add_argument("--full_json", required=True,
                    help="Full dataset JSON (train+test not yet split)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--drop_exclude", action="store_true",
                    help="Also remove not_sure/low_quality samples from the corrected json "
                         "(default: keep them)")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    entries = scan_review_dir(args.review_dir)
    samples, stem_to_idx = load_full(args.full_json)

    # categorize scanned entries
    fakes = [e for e in entries.values() if e["kind"] == "fake"]
    trues = [e for e in entries.values() if e["kind"] == "true"]
    excludes = [e for e in entries.values() if e["kind"] == "exclude"]

    # ---- 1. validate + build fake correction records ----
    corrected_samples = copy.deepcopy(samples)
    fake_records = []
    n_applied = 0
    n_notfound = 0
    n_mismatch = 0
    for e in fakes:
        stem = e["stem"]
        idx = stem_to_idx.get(stem)
        found = idx is not None
        cur_label = None
        applied = False
        if found:
            _, cur_label = get_assistant_and_label(corrected_samples[idx])
            if cur_label != e["orig"]:
                # the json label doesn't match what the review folder assumed
                n_mismatch += 1
            else:
                assistant, _ = get_assistant_and_label(corrected_samples[idx])
                assistant["content"] = str(e["corrected"])
                applied = True
                n_applied += 1
        else:
            n_notfound += 1

        fake_records.append({
            "stem": stem,
            "video_file": e["video_file"],
            "review_category": e["parent_sub"],
            "orig_label": e["orig"],
            "orig_text": LABEL_TEXT[e["orig"]],
            "corrected_label": e["corrected"],
            "corrected_text": LABEL_TEXT[e["corrected"]],
            "found_in_json": found,
            "json_label_before": cur_label,
            "applied": applied,
        })

    # ---- 2. write fake corrections (CSV + JSON) ----
    csv_path = out / "fake_corrections.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fake_records[0].keys()) if fake_records else [])
        w.writeheader()
        for r in fake_records:
            w.writerow(r)
    with open(out / "fake_corrections.json", "w", encoding="utf-8") as f:
        json.dump(fake_records, f, ensure_ascii=False, indent=2)
    print(f"\n  ↳ {csv_path}  ({len(fake_records)} fake records)")
    print(f"  ↳ {out / 'fake_corrections.json'}")

    # ---- 3. compile true hard examples (matched training samples) ----
    def matched_samples(entry_list):
        got = []
        miss = 0
        for e in entry_list:
            idx = stem_to_idx.get(e["stem"])
            if idx is None:
                miss += 1
                continue
            got.append(corrected_samples[idx])
        return got, miss

    true_risk = [e for e in trues if e["orig"] == 1]
    true_safe = [e for e in trues if e["orig"] == 0]
    tr_samples, tr_miss = matched_samples(true_risk)
    ts_samples, ts_miss = matched_samples(true_safe)
    with open(out / "true_risk.json", "w", encoding="utf-8") as f:
        json.dump(tr_samples, f, ensure_ascii=False, indent=2)
    with open(out / "true_safe.json", "w", encoding="utf-8") as f:
        json.dump(ts_samples, f, ensure_ascii=False, indent=2)
    with open(out / "true_hard_all.json", "w", encoding="utf-8") as f:
        json.dump(tr_samples + ts_samples, f, ensure_ascii=False, indent=2)
    print(f"  ↳ {out / 'true_risk.json'}  ({len(tr_samples)} samples, miss {tr_miss})")
    print(f"  ↳ {out / 'true_safe.json'}  ({len(ts_samples)} samples, miss {ts_miss})")
    print(f"  ↳ {out / 'true_hard_all.json'}  ({len(tr_samples)+len(ts_samples)} samples)")

    # ---- 4. compile exclude list ----
    excl_csv = out / "exclude_list.csv"
    with open(excl_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stem", "video_file", "review_category", "orig_label", "found_in_json"])
        for e in excludes:
            w.writerow([e["stem"], e["video_file"], e["parent_sub"], e["orig"],
                        e["stem"] in stem_to_idx])
    print(f"  ↳ {excl_csv}  ({len(excludes)} exclude entries)")

    # ---- 5. write label-corrected full json (optionally drop excludes) ----
    final = corrected_samples
    n_dropped = 0
    if args.drop_exclude:
        excl_stems = {e["stem"] for e in excludes}
        keep = []
        for s in corrected_samples:
            videos = s.get("videos") or []
            stem = Path(videos[0]).stem if videos else None
            if stem in excl_stems:
                n_dropped += 1
                continue
            keep.append(s)
        final = keep

    corrected_json = out / "train_all_label_corrected.json"
    with open(corrected_json, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    # ---- summary ----
    print("\n" + "=" * 70)
    print("📊 Label correction summary")
    print("=" * 70)
    print(f"  fake (flip)     : {len(fakes)}  → applied {n_applied}, "
          f"mismatch {n_mismatch}, not-found {n_notfound}")
    print(f"      fake_risk 1→0: {sum(1 for e in fakes if e['orig']==1)}")
    print(f"      fake_safe 0→1: {sum(1 for e in fakes if e['orig']==0)}")
    print(f"  true (keep)     : {len(trues)}  (true_risk {len(true_risk)}, true_safe {len(true_safe)})")
    print(f"  exclude         : {len(excludes)}"
          + (f"  → DROPPED {n_dropped} from json" if args.drop_exclude else "  (kept in json)"))
    print(f"\n  Corrected full json: {corrected_json}")
    print(f"    size: {len(final)} samples"
          + (f" (was {len(samples)}, dropped {n_dropped})" if args.drop_exclude
             else f" (unchanged count {len(samples)}, only labels flipped)"))
    if n_mismatch:
        print(f"\n  ⚠️  {n_mismatch} fake videos had a json label different from the review "
              f"assumption — NOT flipped. Check fake_corrections.csv (found_in_json / "
              f"json_label_before).")
    print("=" * 70)


if __name__ == "__main__":
    main()
