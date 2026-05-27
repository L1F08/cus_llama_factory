"""Assemble the final dataset: exclude → split → oversample (Path D, step 7+8).

⚠️ LEAKAGE SAFETY: the train/test split happens FIRST, and hard-example
oversampling is applied ONLY to the train side. Hard examples that land in the
test split stay single (no duplicates), and train/test stems are asserted
disjoint. Oversampling before splitting would duplicate a clip across both
sides — this script makes that impossible.

Pipeline:
  1. Load the label-corrected full dataset (fakes already flipped).
  2. Drop any stems in --drop_stems_file (e.g. low-frame / unreadable videos).
  3. Stratified split by label → train_raw, test (--test_ratio).
  4. Oversample ONLY train-side genuine hard examples (stems from
     --true_hard_json) by --oversample_factor.
  5. Write train_final.json + test.json, with a full report.

Usage:
    python build_final_dataset.py \\
        --corrected_json   /.../corrections_v2/train_all_label_corrected.json \\
        --true_hard_json   /.../corrections_v2/true_hard_all.json \\
        --drop_stems_file  /.../low_frame_stems.txt \\
        --out_dir          /.../final_dataset \\
        --test_ratio 0.1 --oversample_factor 3 --seed 42
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def stem_of(sample):
    videos = sample.get("videos") or []
    return Path(videos[0]).stem if videos else None


def label_of(sample):
    a = next(m for m in sample["messages"] if m.get("role") == "assistant")
    raw = str(a["content"]).strip()
    if raw in ("0", "安全"):
        return 0
    if raw in ("1", "高风险"):
        return 1
    return int(raw)


def load_drop_stems(path):
    stems = set()
    if not path:
        return stems
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                stems.add(Path(line).stem)  # accepts a bare stem OR a full path
    return stems


def label_balance(samples):
    c = Counter(label_of(s) for s in samples)
    return c[1], c[0]  # (pos 高风险, neg 安全)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corrected_json", required=True,
                    help="Label-corrected full dataset (from apply_label_corrections.py)")
    ap.add_argument("--true_hard_json", required=True,
                    help="Genuine hard examples to oversample (true_hard_all.json)")
    ap.add_argument("--drop_stems_file", default=None,
                    help="Optional newline list of stems/paths to exclude (e.g. low-frame videos)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--oversample_factor", type=int, default=3,
                    help="Hard examples appear this many times total in TRAIN (1 base + K-1 copies)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_stratify", action="store_true",
                    help="Disable label-stratified split (default: stratified)")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    samples = load_json(args.corrected_json)
    hard_stems = {stem_of(s) for s in load_json(args.true_hard_json)}
    hard_stems.discard(None)
    drop_stems = load_drop_stems(args.drop_stems_file)

    print(f"[load] corrected={len(samples)}  hard={len(hard_stems)}  drop_list={len(drop_stems)}")

    # 2. drop excluded stems
    kept = [s for s in samples if stem_of(s) not in drop_stems]
    n_dropped = len(samples) - len(kept)

    # 3. stratified split by label
    if args.no_stratify:
        random.shuffle(kept)
        n_test = int(round(len(kept) * args.test_ratio))
        test = kept[:n_test]
        train_raw = kept[n_test:]
    else:
        by_label = {0: [], 1: []}
        for s in kept:
            by_label[label_of(s)].append(s)
        train_raw, test = [], []
        for lab, lst in by_label.items():
            random.shuffle(lst)
            n_test = int(round(len(lst) * args.test_ratio))
            test += lst[:n_test]
            train_raw += lst[n_test:]

    # leakage guard
    train_stems = {stem_of(s) for s in train_raw}
    test_stems = {stem_of(s) for s in test}
    overlap = train_stems & test_stems
    assert not overlap, f"LEAKAGE: {len(overlap)} stems in both train and test!"

    # 4. oversample ONLY train-side hard examples
    hard_in_train = [s for s in train_raw if stem_of(s) in hard_stems]
    hard_in_test = [s for s in test if stem_of(s) in hard_stems]
    extra = hard_in_train * max(0, args.oversample_factor - 1)
    train_final = train_raw + extra
    random.shuffle(train_final)

    # 5. write
    with open(out / "train_final.json", "w", encoding="utf-8") as f:
        json.dump(train_final, f, ensure_ascii=False, indent=2)
    with open(out / "test.json", "w", encoding="utf-8") as f:
        json.dump(test, f, ensure_ascii=False, indent=2)

    # report
    tr_pos, tr_neg = label_balance(train_final)
    te_pos, te_neg = label_balance(test)
    raw_pos, raw_neg = label_balance(train_raw)
    print("\n" + "=" * 72)
    print("📦 Final dataset assembled")
    print("=" * 72)
    print(f"  dropped (exclude list)     : {n_dropped}")
    print(f"  after drop                 : {len(kept)}")
    print(f"  ── split (test_ratio={args.test_ratio}, "
          f"{'no-stratify' if args.no_stratify else 'stratified'}) ──")
    print(f"  test                       : {len(test):>6}  (高风险 {te_pos}, 安全 {te_neg})")
    print(f"  train_raw (pre-oversample) : {len(train_raw):>6}  (高风险 {raw_pos}, 安全 {raw_neg})")
    print(f"  ── oversample (factor={args.oversample_factor}, train only) ──")
    print(f"  hard examples in train     : {len(hard_in_train)}  → +{len(extra)} copies")
    print(f"  hard examples in test      : {len(hard_in_test)}  (NOT oversampled, kept single)")
    print(f"  train_final                : {len(train_final):>6}  (高风险 {tr_pos}, 安全 {tr_neg})")
    print(f"\n  ✅ leakage check passed: train ∩ test = ∅")
    print(f"  ↳ {out / 'train_final.json'}")
    print(f"  ↳ {out / 'test.json'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
