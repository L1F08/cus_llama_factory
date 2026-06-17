"""Triage 'invisible-risk' positives via model self-inference (Exp10 pipeline).

Under the leakage-free framing (risk clips end at AEB trigger, -3..0), some
positives don't SHOW the risk yet (side VRU enters view post-trigger; lead
vehicle still far). The model was TRAINED on these clips — if it still
confidently predicts 安全, that is a strong "unlearnable / invisible" signal.

Handling rule (experiments_log 观察12): such samples must be DROPPED —
never label-flipped (the event IS risky: AEB fired), never re-cut to a
post-trigger window (re-introduces the braking leak).

Four modes — two sides, each with triage (build review queue) + apply (clean):

  POSITIVE side (label=高风险, model says 安全 → suspect "risk invisible"):
    --mode triage   queue = positives sorted by P(安全) desc → human sort into
                    visible_risk(keep) / invisible / not_sure / low_quality(drop).
    --mode apply    drop the dropped-verdict stems from the manifest.

  NEGATIVE side (label=安全, model says 高风险 → suspect mislabel OR hard negative):
    --mode triage_neg  queue = negatives sorted by P(高风险) desc → human sort into
                    hard_negative(keep, incl. overtake — the precision moat) /
                    mislabel_risk / not_sure / low_quality(drop).
    --mode apply_neg   drop the dropped-verdict stems.
                    ⚠️ NEVER auto-drop the whole queue: overtake-style hard
                    negatives look identical to mislabels under this probe and
                    MUST be kept; human review is mandatory.

Both sides: labels are never flipped here (drop only). The same apply can run on
training negatives OR on a test set's negatives (to audit test-label noise).
Optional --pred2 second probe: samples both models flag rank first.

Usage:
    # positives (Exp10/13 cleaning)
    python triage_invisible_risk.py --mode triage     --pred pos_pred.json --out_dir out_pos
    python triage_invisible_risk.py --mode apply       --train_json pool.json --review_dir out_pos/review_after --out_json pool_clean.json
    # negatives (training golden, or Test1 negatives audit)
    python triage_invisible_risk.py --mode triage_neg --pred neg_pred.json --out_dir out_neg
    python triage_invisible_risk.py --mode apply_neg   --train_json pool.json --review_dir out_neg/review_after --out_json pool_clean.json
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

# --- POSITIVE side (label=高风险, model says 安全 → suspect "risk invisible") ---
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

# --- NEGATIVE side (label=安全, model says 高风险 → suspect mislabel OR hard negative) ---
# KEEP = genuinely safe (incl. overtake-style hard negatives — the precision moat).
# DROP = mislabeled (actually risky), not_sure, low_quality.
NEG_VERDICT_ACTION = {
    "hard_negative": "keep",  # 真安全但像危险（overtake 类）→ 必须留，精度护城河
    "true_safe": "keep",
    "safe": "keep",
    "keep": "keep",
    "mislabel_risk": "drop",  # 标安全实则危险 → 剔（如确认 -3..0 内可见可手动改标为正样本）
    "real_risk": "drop",
    "risk": "drop",
    "not_sure": "drop",
    "low_quality": "drop",
}

# per-side config: triage 队列按哪个分数排、审核子目录、apply 用哪张 verdict 表
SIDE = {
    "pos": {
        "score_key": "p_safe",
        "title": "正样本（标高风险、模型判安全）→ 疑似风险不可见",
        "both_desc": "两模型都判安全（最强不可学信号）",
        "folders": "visible_risk(留) / invisible(剔) / not_sure(剔) / low_quality(剔)",
        "verdict_map": VERDICT_ACTION,
    },
    "neg": {
        "score_key": "p_risk",
        "title": "负样本（标安全、模型判高风险）→ 疑似标错 或 硬负样本",
        "both_desc": "两模型都判高风险（最强信号）",
        "folders": "hard_negative(真安全/overtake，留) / mislabel_risk(真危险，剔) / not_sure(剔) / low_quality(剔)",
        "verdict_map": NEG_VERDICT_ACTION,
    },
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


# ---------------- mode: triage (pos/neg) ----------------
def run_triage(args, side):
    cfg = SIDE[side]
    sk = cfg["score_key"]            # "p_safe" (pos) or "p_risk" (neg)
    preds = load_pred(args.pred)
    preds2 = load_pred(args.pred2) if args.pred2 else None
    for d in (preds, preds2):
        if d:
            for v in d.values():
                v["p_risk"] = 1.0 - v["p_safe"]

    print(f"\n  triage side = {side}  ——  {cfg['title']}")

    # score distribution over ALL samples of this label
    bins = Counter()
    for v in preds.values():
        bins[min(int(v[sk] * 10), 9)] += 1
    n = len(preds)
    print(f"\n  --- {sk} 分布（全部样本）---")
    for b in range(10):
        cnt = bins.get(b, 0)
        bar = "█" * (cnt * 60 // max(n, 1))
        print(f"  [{b/10:.1f},{(b+1)/10:.1f}) {cnt:>7} {bar}")

    thr = args.p_safe_threshold      # generic queue threshold on the side's score
    queue = [dict(stem=k, **v) for k, v in preds.items() if v[sk] >= thr]
    if preds2 is not None:
        for q in queue:
            v2 = preds2.get(q["stem"])
            q["score2"] = v2[sk] if v2 else None
            q["both"] = bool(v2 and v2[sk] >= thr)
        queue.sort(key=lambda q: (not q["both"], -q[sk]))
        print(f"\n  双探针: 队列中 both（{cfg['both_desc']}）= {sum(1 for q in queue if q['both'])}")
    else:
        queue.sort(key=lambda q: -q[sk])

    for c in (0.9, 0.7, 0.5):
        if c >= thr:
            print(f"  {sk} ≥ {c}: {sum(1 for q in queue if q[sk] >= c)}")
    print(f"  队列总数（{sk}≥{thr}）: {len(queue)} / {n}  ({len(queue)/max(n,1):.1%} 的本类样本)")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # queue csv
    csv_path = out / "triage_queue.csv"
    fields = ["rank", "stem", sk] + (["score2", "both"] if preds2 else []) + \
             ["p_safe", "p_risk", "logit_safe", "logit_risk", "video"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for i, q in enumerate(queue, 1):
            row = [i, q["stem"], f"{q[sk]:.6f}"]
            if preds2:
                row += [f"{q['score2']:.6f}" if q["score2"] is not None else "", int(q["both"])]
            row += [f"{q['p_safe']:.6f}", f"{q['p_risk']:.6f}",
                    q["logit_safe"], q["logit_risk"], q["video"]]
            w.writerow(row)
    print(f"  ↳ {csv_path}")

    # safety guard: an implausibly large queue almost always means wrong input
    # (e.g. feeding the risk pool / positives into triage_neg) — don't copy 30k videos.
    if args.limit and args.limit > 0:
        to_copy = queue[: args.limit]
    elif len(queue) > args.copy_cap:
        print(f"\n  ⚠️⚠️ 队列 {len(queue)} > copy_cap {args.copy_cap}：异常大！")
        print(f"     最可能是输入文件搞错（如 triage_neg 却喂了风险池/正样本推理，"
              f"或 triage 喂了负样本）。")
        print(f"     已跳过视频拷贝（triage_queue.csv 仍写出，可先检查）。"
              f"确认无误后用 --limit N 分批，或 --copy_cap 调大。")
        to_copy = []
    else:
        to_copy = queue

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
    print(f"\n下一步：人工把 review_videos/ 分拣到子目录：{cfg['folders']}，"
          f"然后跑 --mode {'apply' if side == 'pos' else 'apply_neg'}。")


# ---------------- mode: apply ----------------
def label_of(sample):
    a = next(m for m in sample["messages"] if m.get("role") == "assistant")
    raw = str(a["content"]).strip()
    if raw in ("0", "安全"):
        return 0
    if raw in ("1", "高风险"):
        return 1
    return int(raw)


def run_apply(args, side):
    verdict_map = SIDE[side]["verdict_map"]
    # scan verdict folders
    root = Path(args.review_dir)
    if not root.is_dir():
        raise SystemExit(f"❌ review_dir not found: {root}")
    verdicts = {}  # stem -> (verdict, action)
    unknown_dirs = []
    for d in root.rglob("*"):
        if not d.is_dir():
            continue
        action = verdict_map.get(d.name)
        if action is None:
            # a dir that isn't a recognized verdict for this side — flag if it holds videos
            if any(f.suffix.lower() in VIDEO_EXTS for f in d.iterdir() if f.is_file()):
                unknown_dirs.append(d.name)
            continue
        for f in d.iterdir():
            if f.suffix.lower() in VIDEO_EXTS:
                verdicts[f.stem] = (d.name, action)
    if unknown_dirs:
        print(f"  ⚠️ 未识别的子目录（含视频但不在 {side} verdict 表内，已忽略）: {sorted(set(unknown_dirs))}")
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
    ap.add_argument("--mode", required=True,
                    choices=["triage", "apply", "triage_neg", "apply_neg"],
                    help="triage/apply = 正样本（标高风险、判安全）；"
                         "triage_neg/apply_neg = 负样本（标安全、判高风险）")
    # triage
    ap.add_argument("--pred", help="主探针预测 JSON（对该类样本的推理；id=视频路径, logits.{安全,高风险}）")
    ap.add_argument("--pred2", default=None, help="可选第二探针，双探针一致者优先")
    ap.add_argument("--out_dir", help="triage 输出目录")
    ap.add_argument("--p_safe_threshold", type=float, default=0.5,
                    help="队列分数阈值（pos 用 P(安全)、neg 用 P(高风险)），≥ 此值入队（默认 0.5）")
    ap.add_argument("--limit", type=int, default=0, help=">0 时只拷贝队列前 N 个视频")
    ap.add_argument("--copy_cap", type=int, default=5000,
                    help="队列 > 此值且未设 --limit 时跳过拷贝并报警（防喂错文件，默认 5000）")
    # apply
    ap.add_argument("--train_json", help="要清洗的 manifest（含正负样本；也可只是训练负样本/Test1集）")
    ap.add_argument("--review_dir", help="人工分拣后的目录（含 verdict 子目录）")
    ap.add_argument("--out_json", help="清洗后输出路径")
    args = ap.parse_args()

    side = "neg" if args.mode.endswith("neg") else "pos"
    if args.mode in ("triage", "triage_neg"):
        if not args.pred or not args.out_dir:
            raise SystemExit("❌ triage 需要 --pred 与 --out_dir")
        run_triage(args, side)
    else:
        if not (args.train_json and args.review_dir and args.out_json):
            raise SystemExit("❌ apply 需要 --train_json / --review_dir / --out_json")
        run_apply(args, side)


if __name__ == "__main__":
    main()
