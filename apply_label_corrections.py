"""Apply manual-review results back to the full dataset (Path D, step 2).

Supports MULTIPLE review rounds and BOTH folder layouts. Each review directory
is scanned RECURSIVELY for verdict subfolders, by folder NAME:

    true_risk / true_safe    标签对 → 保留 (keep the json label)
    fake_risk / fake_safe     标签错 → 翻转 (flip the json label)
    not_sure / low_quality    不适合训练 → exclude 候选

The original label is ALWAYS read from the dataset json (ground truth), NOT
inferred from folder nesting. So both layouts work uniformly:

    suspect_videos_after_check/label1_risk/true_risk/...   (round 1, nested)
    hard_fn_to_review_after_check/true_risk/...            (round 2, flat)

You can pass several review dirs at once and they are merged (dedup by stem).

This script:
  1. Recursively scans the review dir(s) → verdict per video stem.
  2. Resolves each video's original label from the full json.
  3. Flips the "fake" ones, writes a correction audit (CSV + JSON).
  4. Produces a label-corrected copy of the FULL dataset json.
  5. Compiles the genuine hard examples (true → split into risk/safe by the
     json label) for a later oversampling decision (factor NOT fixed here).
  6. Compiles the exclude list (not_sure + low_quality); removed from the
     corrected json only if --drop_exclude is given.

Label convention: assistant content "0" = 安全, "1" = 高风险.
Matching between folders and dataset is by video-path stem (basename w/o ext).

Usage:
    python apply_label_corrections.py \\
        --review_dir DIR1 [DIR2 ...] \\
        --full_json  /.../train_all_..._cleaned.json \\
        --out_dir    /.../corrections \\
        [--drop_exclude]
"""

import argparse
import copy
import csv
import json
from pathlib import Path

LABEL_TEXT = {0: "安全", 1: "高风险"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# verdict folder name -> kind. "true"=keep json label, "fake"=flip, "exclude"=drop-candidate.
VERDICT_KIND = {
    "true_risk":   "true",
    "true_safe":   "true",
    "fake_risk":   "fake",
    "fake_safe":   "fake",
    "not_sure":    "exclude",
    "low_quality": "exclude",
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


def scan_review_dirs(review_dirs):
    """Recursively find verdict subfolders across all review dirs.

    Returns stem -> {stem, verdict, kind, video_file}. Original label is NOT
    taken from the folder — it is resolved later from the dataset json.
    """
    entries = {}
    dupes = []
    for review_dir in review_dirs:
        root = Path(review_dir)
        if not root.is_dir():
            print(f"⚠️  review_dir not found, skipping: {review_dir}")
            continue
        # find every directory whose name is a known verdict (at any depth)
        for d in root.rglob("*"):
            if not d.is_dir():
                continue
            kind = VERDICT_KIND.get(d.name)
            if kind is None:
                continue
            for f in d.iterdir():
                if f.suffix.lower() not in VIDEO_EXTS:
                    continue
                stem = f.stem
                if stem in entries:
                    dupes.append((stem, entries[stem]["verdict"], d.name))
                    continue
                entries[stem] = {
                    "stem": stem,
                    "verdict": d.name,
                    "kind": kind,
                    "video_file": str(f),
                }
    print(f"[review] scanned {len(review_dirs)} dir(s) → {len(entries)} unique video verdicts")
    if dupes:
        print(f"⚠️  {len(dupes)} stems appeared in multiple verdict folders (kept first):")
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
    ap.add_argument("--review_dir", required=True, nargs="+",
                    help="One or more review dirs (scanned recursively for verdict subfolders)")
    ap.add_argument("--full_json", required=True,
                    help="Full dataset JSON (train+test not yet split)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--drop_exclude", action="store_true",
                    help="Also remove not_sure/low_quality samples from the corrected json "
                         "(default: keep them)")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    entries = scan_review_dirs(args.review_dir)
    samples, stem_to_idx = load_full(args.full_json)

    corrected_samples = copy.deepcopy(samples)

    fake_records = []
    true_entries = []     # (entry, orig_label)
    exclude_entries = []  # entry
    n_fake_applied = n_fake_notfound = 0
    n_true = n_true_notfound = 0
    n_excl = 0

    for e in entries.values():
        stem = e["stem"]
        idx = stem_to_idx.get(stem)
        found = idx is not None
        orig = None
        if found:
            _, orig = get_assistant_and_label(corrected_samples[idx])

        if e["kind"] == "fake":
            corrected = (1 - orig) if found else None
            applied = False
            if found:
                assistant, _ = get_assistant_and_label(corrected_samples[idx])
                assistant["content"] = str(corrected)
                applied = True
                n_fake_applied += 1
            else:
                n_fake_notfound += 1
            fake_records.append({
                "stem": stem,
                "video_file": e["video_file"],
                "verdict": e["verdict"],
                "found_in_json": found,
                "orig_label": orig,
                "orig_text": LABEL_TEXT.get(orig),
                "corrected_label": corrected,
                "corrected_text": LABEL_TEXT.get(corrected),
                "applied": applied,
            })

        elif e["kind"] == "true":
            if found:
                true_entries.append((e, orig))
                n_true += 1
            else:
                n_true_notfound += 1

        else:  # exclude
            exclude_entries.append(e)
            n_excl += 1

    # ---- write fake corrections (CSV + JSON) ----
    csv_path = out / "fake_corrections.csv"
    fields = ["stem", "video_file", "verdict", "found_in_json", "orig_label",
              "orig_text", "corrected_label", "corrected_text", "applied"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in fake_records:
            w.writerow(r)
    with open(out / "fake_corrections.json", "w", encoding="utf-8") as f:
        json.dump(fake_records, f, ensure_ascii=False, indent=2)
    print(f"\n  ↳ {csv_path}  ({len(fake_records)} fake records)")
    print(f"  ↳ {out / 'fake_corrections.json'}")

    # ---- compile true hard examples (use CORRECTED samples; split by json label) ----
    true_risk = [corrected_samples[stem_to_idx[e['stem']]] for e, o in true_entries if o == 1]
    true_safe = [corrected_samples[stem_to_idx[e['stem']]] for e, o in true_entries if o == 0]
    with open(out / "true_risk.json", "w", encoding="utf-8") as f:
        json.dump(true_risk, f, ensure_ascii=False, indent=2)
    with open(out / "true_safe.json", "w", encoding="utf-8") as f:
        json.dump(true_safe, f, ensure_ascii=False, indent=2)
    with open(out / "true_hard_all.json", "w", encoding="utf-8") as f:
        json.dump(true_risk + true_safe, f, ensure_ascii=False, indent=2)
    print(f"  ↳ {out / 'true_risk.json'}  ({len(true_risk)} samples)")
    print(f"  ↳ {out / 'true_safe.json'}  ({len(true_safe)} samples)")
    print(f"  ↳ {out / 'true_hard_all.json'}  ({len(true_risk)+len(true_safe)} samples)")

    # ---- exclude list ----
    excl_csv = out / "exclude_list.csv"
    with open(excl_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stem", "video_file", "verdict", "found_in_json"])
        for e in exclude_entries:
            w.writerow([e["stem"], e["video_file"], e["verdict"], e["stem"] in stem_to_idx])
    print(f"  ↳ {excl_csv}  ({len(exclude_entries)} exclude entries)")

    # ---- label-corrected full json (optionally drop excludes) ----
    final = corrected_samples
    n_dropped = 0
    if args.drop_exclude:
        excl_stems = {e["stem"] for e in exclude_entries}
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
    fr1 = sum(1 for r in fake_records if r["orig_label"] == 1)
    fr0 = sum(1 for r in fake_records if r["orig_label"] == 0)
    print(f"  fake (flip)  : {len(fake_records)}  → applied {n_fake_applied}, "
          f"not-found {n_fake_notfound}")
    print(f"      orig 高风险1→安全0 : {fr1}")
    print(f"      orig 安全0→高风险1 : {fr0}")
    print(f"  true (keep)  : {n_true}  (true_risk {len(true_risk)}, true_safe {len(true_safe)}"
          + (f", not-found {n_true_notfound}" if n_true_notfound else "") + ")")
    print(f"  exclude      : {n_excl}"
          + (f"  → DROPPED {n_dropped} from json" if args.drop_exclude else "  (kept in json)"))
    print(f"\n  Corrected full json: {corrected_json}")
    print(f"    size: {len(final)} samples"
          + (f" (was {len(samples)}, dropped {n_dropped})" if args.drop_exclude
             else f" (unchanged count {len(samples)}, only labels flipped)"))
    print("=" * 70)


if __name__ == "__main__":
    main()
