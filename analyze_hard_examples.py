"""Hard-example mining for the collision-risk training set (Path D).

Matches model predictions (inference run on the training set, in test format)
against the labeled training manifest, computes per-sample difficulty
(cross-entropy loss), categorizes every sample, and exports hard-example
subsets in the ORIGINAL training format so they can be fed straight back
into LlamaFactory for enhancement training.

Inputs
------
--pred   : prediction JSON, list of
           {"id": "<video_path>", "answers": ["高风险"|"安全"],
            "logits": {"安全": float, "高风险": float}}
--train  : labeled training manifest, list of
           {"messages": [{"role":"user","content":...},
                         {"role":"assistant","content":"0"|"1"}],
            "videos": ["<video_path>"]}
           where assistant content "0" = 安全 (neg), "1" = 高风险 (pos).

Matching is done on the video path STEM (basename without extension), so
absolute-path differences between the two files don't matter.

Outputs (written to --out_dir, all in the original training JSON-array format
so they are directly trainable):
    hard_fn.json            真实高风险但被预测为安全（漏报）—— FN，最高优先级
    hard_fp.json            真实安全但被预测为高风险（误报）—— FP
    borderline.json         模型不确定的样本 (|P-0.5| < --borderline_band)
    hard_all.json           借鉴 OHEM：borderline ∪ 所有错例（去重，去掉疑似噪声）
    suspect_label0_safe.json  高置信度错误中、原标签=安全(0) 的（模型→高风险）
    suspect_label1_risk.json  高置信度错误中、原标签=高风险(1) 的（模型→安全, ≈FN）
    augmented_train.json     (可选) 原训练集 + hard_all × K，用于增强训练
    per_sample.csv           每条样本的 id/label/P/loss/category，便于自定义分析

可选：--suspect_video_dir 会把疑似标签噪声的视频原文件，按原标签分两个子目录
复制出来，方便直接观看抽查（判断是真·难例还是标注错误）：
    <suspect_video_dir>/label0_safe/   原标签安全、模型误判高风险
    <suspect_video_dir>/label1_risk/   原标签高风险、模型误判安全（≈FN）
"""

import argparse
import csv
import json
import math
import os
import shutil
from pathlib import Path

EPS = 1e-12


def softmax_p_risk(logit_safe: float, logit_risk: float) -> float:
    """P(高风险) via 2-class softmax (numerically stable)."""
    m = max(logit_safe, logit_risk)
    es = math.exp(logit_safe - m)
    er = math.exp(logit_risk - m)
    return er / (es + er)


def load_predictions(pred_path: str) -> dict:
    """stem -> dict(P, logit_safe, logit_risk, pred_text)."""
    with open(pred_path, "r", encoding="utf-8") as f:
        preds = json.load(f)

    out = {}
    skipped = 0
    for p in preds:
        logits = p.get("logits")
        if not logits or "安全" not in logits or "高风险" not in logits:
            skipped += 1
            continue
        try:
            ls = float(logits["安全"])
            lr = float(logits["高风险"])
        except (TypeError, ValueError):
            skipped += 1
            continue
        stem = Path(p["id"]).stem
        out[stem] = {
            "P": softmax_p_risk(ls, lr),
            "logit_safe": ls,
            "logit_risk": lr,
            "pred_text": (p.get("answers") or [""])[0],
        }
    print(f"[pred ] {pred_path}")
    print(f"        usable={len(out)} (skipped {skipped} without valid logits)")
    return out


def load_training(train_path: str) -> dict:
    """stem -> dict(label, raw_sample)."""
    with open(train_path, "r", encoding="utf-8") as f:
        train = json.load(f)

    out = {}
    bad = 0
    for s in train:
        try:
            videos = s.get("videos") or []
            if not videos:
                bad += 1
                continue
            stem = Path(videos[0]).stem
            # label is the assistant turn content: "0" (安全) or "1" (高风险)
            assistant = next(m for m in s["messages"] if m.get("role") == "assistant")
            raw = str(assistant["content"]).strip()
            if raw in ("0", "安全"):
                label = 0
            elif raw in ("1", "高风险"):
                label = 1
            else:
                label = int(raw)
            out[stem] = {"label": label, "raw_sample": s}
        except (KeyError, StopIteration, ValueError):
            bad += 1
            continue
    print(f"[train] {train_path}")
    print(f"        usable={len(out)} (skipped {bad} malformed)")
    return out


def analyze(preds: dict, train: dict, threshold: float,
            borderline_band: float, high_conf_band: float):
    """Join on stem, compute per-sample metrics + category."""
    rows = []
    matched = 0
    for stem, tr in train.items():
        if stem not in preds:
            continue
        matched += 1
        pr = preds[stem]
        y = tr["label"]
        P = pr["P"]
        pred = 1 if P >= threshold else 0
        correct = (pred == y)
        margin = abs(P - 0.5)

        # per-sample cross-entropy loss = hardness (OHEM/focal-style)
        p_true = P if y == 1 else (1.0 - P)
        ce_loss = -math.log(p_true + EPS)

        # confusion category
        if y == 1 and pred == 1:
            conf_cat = "TP"
        elif y == 0 and pred == 0:
            conf_cat = "TN"
        elif y == 0 and pred == 1:
            conf_cat = "FP"
        else:
            conf_cat = "FN"

        # confidence band
        if margin < borderline_band:
            band = "borderline"
        elif margin >= high_conf_band:
            band = "high_conf"
        else:
            band = "mid"

        rows.append({
            "stem": stem,
            "label": y,
            "P": P,
            "pred": pred,
            "correct": correct,
            "margin": margin,
            "ce_loss": ce_loss,
            "conf_cat": conf_cat,
            "band": band,
            "raw_sample": tr["raw_sample"],
        })

    print(f"[match] matched {matched} / {len(train)} training samples "
          f"({len(train) - matched} had no prediction)")
    return rows


def print_summary(rows, threshold):
    n = len(rows)
    if n == 0:
        print("❌ No matched samples.")
        return

    cm = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    for r in rows:
        cm[r["conf_cat"]] += 1

    pos = cm["TP"] + cm["FN"]
    neg = cm["TN"] + cm["FP"]
    acc = (cm["TP"] + cm["TN"]) / n
    prec = cm["TP"] / (cm["TP"] + cm["FP"]) if (cm["TP"] + cm["FP"]) else 0.0
    rec = cm["TP"] / (cm["TP"] + cm["FN"]) if (cm["TP"] + cm["FN"]) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    print("\n" + "=" * 78)
    print(f"📊 Training-set analysis @ threshold {threshold} | N={n} "
          f"(pos={pos}, neg={neg})")
    print("=" * 78)
    print(f"  Accuracy {acc:.4f} | Precision {prec:.4f} | Recall {rec:.4f} | F1 {f1:.4f}")
    print(f"  Confusion:  TP={cm['TP']}  TN={cm['TN']}  FP={cm['FP']}  FN={cm['FN']}")

    # accuracy by confidence band — where does the model actually fail?
    print("\n  --- Accuracy by confidence band ---")
    print(f"  {'band':<12}|{'count':>7}|{'correct':>9}|{'acc':>8}|{'avg loss':>10}")
    print("  " + "-" * 50)
    for band in ("high_conf", "mid", "borderline"):
        sub = [r for r in rows if r["band"] == band]
        if not sub:
            continue
        c = sum(r["correct"] for r in sub)
        avg_loss = sum(r["ce_loss"] for r in sub) / len(sub)
        print(f"  {band:<12}|{len(sub):>7}|{c:>9}|{c/len(sub):>8.4f}|{avg_loss:>10.4f}")

    # P distribution histogram (text)
    print("\n  --- P(高风险) distribution ---")
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        cnt = sum(1 for r in rows if lo <= r["P"] < hi)
        bar = "█" * (cnt * 60 // n) if n else ""
        print(f"  [{lo:.1f},{min(hi,1.0):.1f}) {cnt:>6} {bar}")
    print("=" * 78)


def dump_json(samples, path):
    """Write list of raw training samples as a JSON array (original format)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump([s["raw_sample"] for s in samples], f, ensure_ascii=False, indent=2)
    print(f"  ↳ {path}  ({len(samples)} samples)")


def copy_videos(samples, dest_dir):
    """Copy each sample's video (raw_sample['videos'][0]) into dest_dir."""
    os.makedirs(dest_dir, exist_ok=True)
    copied, missing, failed = 0, 0, 0
    for s in samples:
        videos = s["raw_sample"].get("videos") or []
        if not videos:
            continue
        src = videos[0]
        if not os.path.exists(src):
            missing += 1
            continue
        try:
            shutil.copy2(src, dest_dir)
            copied += 1
        except Exception as e:
            failed += 1
            print(f"  ❌ copy failed [{Path(src).name}]: {e}")
    print(f"  ↳ copied {copied} suspect videos to {dest_dir} "
          f"(missing {missing}, failed {failed})")


def export_hard_sets(rows, out_dir, borderline_band, high_conf_band,
                     oversample_factor, suspect_video_dir=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fn = [r for r in rows if r["conf_cat"] == "FN"]
    fp = [r for r in rows if r["conf_cat"] == "FP"]
    borderline = [r for r in rows if r["band"] == "borderline"]
    # suspect label noise = wrong AND very confident
    suspect = [r for r in rows
               if (not r["correct"]) and r["margin"] >= high_conf_band]

    # hard_all = borderline ∪ all errors, MINUS suspected label noise,
    # deduplicated by stem (sorted hardest-first by ce_loss)
    suspect_stems = {r["stem"] for r in suspect}
    hard_map = {}
    for r in borderline + fn + fp:
        if r["stem"] in suspect_stems:
            continue
        hard_map[r["stem"]] = r
    hard_all = sorted(hard_map.values(), key=lambda r: -r["ce_loss"])

    print("\n" + "=" * 78)
    print("📦 Exporting hard-example sets (original training format):")
    print("=" * 78)
    dump_json(sorted(fn, key=lambda r: -r["ce_loss"]), out / "hard_fn.json")
    dump_json(sorted(fp, key=lambda r: -r["ce_loss"]), out / "hard_fp.json")
    dump_json(sorted(borderline, key=lambda r: -r["ce_loss"]), out / "borderline.json")
    dump_json(hard_all, out / "hard_all.json")

    # split suspected label-noise by ORIGINAL label for separate review
    #   label 0 (安全):   标注为安全，但模型高置信预测高风险
    #   label 1 (高风险): 标注为高风险，但模型高置信预测安全 —— 对应 FN，最危险
    suspect_l0 = sorted([r for r in suspect if r["label"] == 0],
                        key=lambda r: -r["ce_loss"])
    suspect_l1 = sorted([r for r in suspect if r["label"] == 1],
                        key=lambda r: -r["ce_loss"])
    dump_json(suspect_l0, out / "suspect_label0_safe.json")
    dump_json(suspect_l1, out / "suspect_label1_risk.json")

    # copy suspect videos out for manual spot-check, split by original label
    if suspect_video_dir:
        copy_videos(suspect_l0, os.path.join(suspect_video_dir, "label0_safe"))
        copy_videos(suspect_l1, os.path.join(suspect_video_dir, "label1_risk"))

    # per-sample CSV
    csv_path = out / "per_sample.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stem", "label", "P", "pred", "correct", "margin",
                    "ce_loss", "conf_cat", "band"])
        for r in sorted(rows, key=lambda r: -r["ce_loss"]):
            w.writerow([r["stem"], r["label"], f"{r['P']:.6f}", r["pred"],
                        int(r["correct"]), f"{r['margin']:.6f}",
                        f"{r['ce_loss']:.6f}", r["conf_cat"], r["band"]])
    print(f"  ↳ {csv_path}  (all {len(rows)} samples, sorted hardest-first)")

    # optional augmented training manifest
    if oversample_factor and oversample_factor > 1:
        all_samples = [r["raw_sample"] for r in rows]
        hard_samples = [r["raw_sample"] for r in hard_all]
        augmented = all_samples + hard_samples * (oversample_factor - 1)
        aug_path = out / f"augmented_train_x{oversample_factor}.json"
        with open(aug_path, "w", encoding="utf-8") as f:
            json.dump(augmented, f, ensure_ascii=False, indent=2)
        print(f"  ↳ {aug_path}  (orig {len(all_samples)} + hard×{oversample_factor-1} "
              f"= {len(augmented)} samples)")

    print("\n  Summary of hard sets:")
    print(f"    FN (漏报, 最高优先级):        {len(fn)}")
    print(f"    FP (误报):                    {len(fp)}")
    print(f"    Borderline (模型不确定):      {len(borderline)}")
    print(f"    hard_all (训练增强用, 去噪后): {len(hard_all)}")
    print(f"    suspect 原标签0-安全 (模型→高风险): {len(suspect_l0)}")
    print(f"    suspect 原标签1-高风险 (模型→安全, ≈FN): {len(suspect_l1)}")
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True,
                    help="Prediction JSON (inference run on the training set)")
    ap.add_argument("--train", required=True,
                    help="Labeled training manifest JSON")
    ap.add_argument("--out_dir", default="./hard_examples")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Decision threshold for P(高风险) (default 0.5)")
    ap.add_argument("--borderline_band", type=float, default=0.15,
                    help="|P-0.5| < this → borderline / uncertain (default 0.15 → P in 0.35~0.65)")
    ap.add_argument("--high_conf_band", type=float, default=0.4,
                    help="|P-0.5| >= this → high confidence (default 0.4 → P<0.1 or >0.9)")
    ap.add_argument("--oversample_factor", type=int, default=0,
                    help="If >1, emit augmented_train manifest = orig + hard_all×(factor-1)")
    ap.add_argument("--suspect_video_dir", default=None,
                    help="If set, copy suspect_label_noise videos into this dir for manual review")
    args = ap.parse_args()

    preds = load_predictions(args.pred)
    train = load_training(args.train)
    rows = analyze(preds, train, args.threshold,
                   args.borderline_band, args.high_conf_band)
    print_summary(rows, args.threshold)
    export_hard_sets(rows, args.out_dir, args.borderline_band,
                     args.high_conf_band, args.oversample_factor,
                     suspect_video_dir=args.suspect_video_dir)


if __name__ == "__main__":
    main()
