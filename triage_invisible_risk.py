"""Triage 'invisible-risk' positives via model self-inference (Exp10 pipeline).

Under the leakage-free framing (risk clips end at AEB trigger, -3..0), some
positives don't SHOW the risk yet (side VRU enters view post-trigger; lead
vehicle still far). The model was TRAINED on these clips — if it still
confidently predicts 安全, that is a strong "unlearnable / invisible" signal.

Handling rule (experiments_log 观察12): such samples must be DROPPED —
never label-flipped (the event IS risky: AEB fired), never re-cut to a
post-trigger window (re-introduces the braking leak).

Two modes:

  --mode triage   Input: prediction JSON over ALL label=1 training positives
                  (id = video path, logits.{安全,高风险}).
                  Output: suspect queue sorted by P(安全) desc, P(安全)
                  distribution stats, triage_queue.csv, and copies of the
                  queue videos into <out_dir>/review_videos/ for human
                  sorting into subfolders:
                      visible_risk/   风险在窗口内可见   → keep
                      invisible/      风险不可见         → drop
                      not_sure/       说不清            → drop
                      low_quality/    视频质量差         → drop
                  Optional --pred2 (e.g. Exp6 as a second probe): samples
                  where BOTH models say 安全 rank first (strongest signal).

  --mode apply    Input: full train manifest + the human-sorted review dir.
                  Output: cleaned train JSON (drop stems sorted into
                  invisible/not_sure/low_quality; keep everything else;
                  labels untouched).

Usage:
    # step 1: build queue + copy videos for review
    python triage_invisible_risk.py --mode triage \\
        --pred /path/to/positives_pred.json \\
        [--pred2 /path/to/exp6_pred.json] \\
        --out_dir /path/to/triage_out \\
        [--p_safe_threshold 0.5] [--limit 0]

    # step 3: after human sorting, produce the cleaned train set
    python triage_invisible_risk.py --mode apply \\
        --train_json /path/to/train_47k_-3_0.json \\
        --review_dir /path/to/triage_out/review_videos_after_check \\
        --out_json   /path/to/train_47k_-3_0_cleaned.json
"""

import argparse
import csv
import json
import math
import os
import shutil
from collections import Counter
from pathlib import Path

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# verdict folder name -> action
VERDICT_ACTION = {
    "visible_risk": "keep",
    "visible": "keep",
    "true_risk": "keep",      # alias, consistent with earlier review rounds
    "invisible": "drop",
    "fake_risk": "drop",      # alias: if reviewer decides label itself is wrong, still drop (no flips here)
    "not_sure": "drop",
    "low_quality": "drop",
}


def p_safe_of(logits: dict):
    try:
        ls = float(logits["安全"])
        lr = float(logits["高风险"])
    except (KeyError, TypeError, ValueError):
        return None
    m = max(ls, lr)
    es, er = math.exp(ls - m), math.exp(lr - m)
    return es / (es + er)


def load_pred(path: str) -> dict:
    """stem -> {p_safe, logit_safe, logit_risk, video}"""
    with open(path, "r", encoding="utf-8") as f:
        preds = json.load(f)
    out, skipped = {}, 0
    for p in preds:
        ps = p_safe_of(p.get("logits") or {})
        if ps is None:
            skipped += 1
            continue
        stem = Path(p["id"]).stem
        out[stem] = {
            "p_safe": ps,
            "logit_safe": float(p["logits"]["安全"]),
            "logit_risk": float(p["logits"]["高风险"]),
            "video": p["id"],
        }
    print(f"[pred ] {path}: usable={len(out)} (skipped {skipped} without valid logits)")
    return out


# ---------------- mode: triage ----------------
def run_triage(args):
    preds = load_pred(args.pred)
    preds2 = load_pred(args.pred2) if args.pred2 else None

    # P(安全) distribution over ALL positives
    bins = Counter()
    for v in preds.values():
        bins[min(int(v["p_safe"] * 10), 9)] += 1
    n = len(preds)
    print("\n  --- P(安全) 分布（全部正样本）---")
    for b in range(10):
        cnt = bins.get(b, 0)
        bar = "█" * (cnt * 60 // max(n, 1))
        print(f"  [{b/10:.1f},{(b+1)/10:.1f}) {cnt:>7} {bar}")

    thr = args.p_safe_threshold
    queue = [dict(stem=k, **v) for k, v in preds.items() if v["p_safe"] >= thr]
    if preds2 is not None:
        for q in queue:
            v2 = preds2.get(q["stem"])
            q["p_safe2"] = v2["p_safe"] if v2 else None
            q["both_safe"] = bool(v2 and v2["p_safe"] >= thr)
        queue.sort(key=lambda q: (not q["both_safe"], -q["p_safe"]))
        n_both = sum(1 for q in queue if q["both_safe"])
        print(f"\n  双探针: 队列中 both-safe（两模型都判安全，最强不可学信号）= {n_both}")
    else:
        queue.sort(key=lambda q: -q["p_safe"])

    for c in (0.9, 0.7, 0.5):
        if c >= thr:
            print(f"  P(安全) ≥ {c}: {sum(1 for q in queue if q['p_safe'] >= c)}")
    print(f"  队列总数（≥{thr}）: {len(queue)} / {n}  "
          f"({len(queue)/max(n,1):.1%} 的正样本)")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # queue csv
    csv_path = out / "triage_queue.csv"
    fields = ["rank", "stem", "p_safe"] + (["p_safe2", "both_safe"] if preds2 else []) + \
             ["logit_safe", "logit_risk", "video"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for i, q in enumerate(queue, 1):
            row = [i, q["stem"], f"{q['p_safe']:.6f}"]
            if preds2:
                row += [f"{q['p_safe2']:.6f}" if q["p_safe2"] is not None else "",
                        int(q["both_safe"])]
            row += [q["logit_safe"], q["logit_risk"], q["video"]]
            w.writerow(row)
    print(f"  ↳ {csv_path}")

    # copy videos for review
    to_copy = queue[: args.limit] if args.limit and args.limit > 0 else queue
    vid_dir = out / "review_videos"
    vid_dir.mkdir(exist_ok=True)
    copied = missing = 0
    for q in to_copy:
        if os.path.exists(q["video"]):
            shutil.copy2(q["video"], vid_dir)
            copied += 1
        else:
            missing += 1
    print(f"  ↳ {vid_dir}: copied {copied}, missing {missing}"
          + (f" (limit={args.limit})" if args.limit else ""))
    print("\n下一步：人工把 review_videos/ 里的视频分拣到子目录 "
          "visible_risk / invisible / not_sure / low_quality，然后跑 --mode apply。")


# ---------------- mode: apply ----------------
def label_of(sample):
    a = next(m for m in sample["messages"] if m.get("role") == "assistant")
    raw = str(a["content"]).strip()
    if raw in ("0", "安全"):
        return 0
    if raw in ("1", "高风险"):
        return 1
    return int(raw)


def run_apply(args):
    # scan verdict folders
    root = Path(args.review_dir)
    if not root.is_dir():
        raise SystemExit(f"❌ review_dir not found: {root}")
    verdicts = {}  # stem -> (verdict, action)
    for d in root.rglob("*"):
        if not d.is_dir():
            continue
        action = VERDICT_ACTION.get(d.name)
        if action is None:
            continue
        for f in d.iterdir():
            if f.suffix.lower() in VIDEO_EXTS:
                verdicts[f.stem] = (d.name, action)
    cnt = Counter(v[0] for v in verdicts.values())
    print(f"[review] {len(verdicts)} sorted videos: "
          + ", ".join(f"{k}={v}" for k, v in sorted(cnt.items())))
    drop_stems = {s for s, (_, a) in verdicts.items() if a == "drop"}

    with open(args.train_json, "r", encoding="utf-8") as f:
        train = json.load(f)

    kept, dropped, drop_not_found = [], 0, set(drop_stems)
    for s in train:
        videos = s.get("videos") or []
        stem = Path(videos[0]).stem if videos else None
        if stem in drop_stems:
            dropped += 1
            drop_not_found.discard(stem)
        else:
            kept.append(s)

    pos = sum(1 for s in kept if label_of(s) == 1)
    neg = len(kept) - pos
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 64)
    print("📦 Cleaned training set")
    print("=" * 64)
    print(f"  原始: {len(train)}  → 剔除 {dropped}  → 剩余 {len(kept)}")
    print(f"  剩余 正(高风险) {pos} : 负(安全) {neg}  = {pos/max(neg,1):.2f}:1")
    if drop_not_found:
        print(f"  ⚠️ {len(drop_not_found)} 个 drop 判定的 stem 在 train json 里没找到（抽样: "
              f"{list(drop_not_found)[:5]}）")
    print(f"  ↳ {args.out_json}")
    print("  注意：只做了剔除，没有任何标签翻转/窗口改动。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["triage", "apply"], required=True)
    # triage
    ap.add_argument("--pred", help="主探针预测 JSON（Exp8 对全部高风险正样本的推理）")
    ap.add_argument("--pred2", default=None, help="可选第二探针（如 Exp6），both-safe 优先")
    ap.add_argument("--out_dir", help="triage 输出目录")
    ap.add_argument("--p_safe_threshold", type=float, default=0.5,
                    help="P(安全) ≥ 此值进入审核队列（默认 0.5）")
    ap.add_argument("--limit", type=int, default=0, help=">0 时只拷贝队列前 N 个视频")
    # apply
    ap.add_argument("--train_json", help="完整训练 manifest（-3..0 全集，含正负）")
    ap.add_argument("--review_dir", help="人工分拣后的目录（含 visible_risk/invisible/... 子目录）")
    ap.add_argument("--out_json", help="清洗后训练集输出路径")
    args = ap.parse_args()

    if args.mode == "triage":
        if not args.pred or not args.out_dir:
            raise SystemExit("❌ triage 需要 --pred 与 --out_dir")
        run_triage(args)
    else:
        if not (args.train_json and args.review_dir and args.out_json):
            raise SystemExit("❌ apply 需要 --train_json / --review_dir / --out_json")
        run_apply(args)


if __name__ == "__main__":
    main()
