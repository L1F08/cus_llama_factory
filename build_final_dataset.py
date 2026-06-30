"""Assemble the final dataset with explicit label-count control (Path D, step 7+8).

Samples EXACT counts of each label into train and test (no ratio), guarantees
train/test are disjoint (test reserved first, train drawn from the remainder),
and emits three files in the required formats:

  train_final.json  — LlamaFactory ShareGPT:
      {"messages": [{"role":"user","content": <PROMPT with <video>>},
                    {"role":"assistant","content": "安全"|"高风险"}],
       "videos": ["<path>"]}

  test.json         — inference input (list of [user-message]):
      [ [ {"role":"user","content":[
              {"type":"video","video":"<path>"},
              {"type":"text","text": <PROMPT without <video>>}]} ], ... ]

  test.jsonl        — GT + content (one json per line):
      {"id":"<stem>","label":0|1,"content":[
          {"type":"text","text": <prompt text before <video>>},
          {"type":"video","video":"<path>","max_pixels":336000},
          {"type":"text","text": <prompt text after <video>>}]}

⚠️ LEAKAGE SAFETY: test samples are reserved first, train drawn from what's
left; train/test stems are asserted disjoint. Optional hard-example
oversampling (--true_hard_json + --oversample_factor) applies to TRAIN ONLY,
on top of the requested train counts.

Usage:
    python build_final_dataset.py \\
        --corrected_json  /.../train_all_label_corrected.json \\
        --drop_stems_file /.../low_frame_stems.txt \\
        --out_dir         /.../final_dataset \\
        --train_pos 18000 --train_neg 18000 \\
        --test_pos 2000  --test_neg 2000 \\
        [--true_hard_json /.../true_hard_all.json --oversample_factor 3] \\
        [--max_pixels 336000] [--seed 42]
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

# Shared train/test prompt. Each <video> marks where a clip goes; the order of
# <video> tokens MUST match the order of paths in each sample's "videos" list.
# 3-cam version: front + left-fisheye + right-fisheye, braking-clause REMOVED
# (no post-trigger braking in -3..0). ⚠️ ORDER ASSUMPTION = [front, left, right];
# verify against your data pipeline (a verification print of sample[0] is emitted).
PROMPT = (
    "你是一个自动驾驶安全专家。请观看以下同一时刻、同一车辆的三路行车视频：\n"
    "自车前视视角<video>\n"
    "自车左侧鱼眼视角<video>\n"
    "自车右侧鱼眼视角<video>\n"
    "任务：自动驾驶碰撞风险二分类。\n"
    "请综合三个视角，判断当前自车是否面临真实的碰撞风险。\n\n"
    "【判定规则】：\n"
    "- 输出“高风险”（真实危险）：自车行驶轨迹上存在即将发生物理碰撞的实体威胁，"
    "包括但不限于以下典型场景：\n"
    "  1. 绝对距离压迫：正前方已有明确的实体障碍物（或静止目标）极度逼近。\n"
    "  2. 纵向追尾/相对速度危险：自车车速过快或前方目标骤停，相对距离急速缩短。\n"
    "  3. 横向/盲区突发侵入：左右鱼眼视角中，行人/两轮车/其他车辆等从侧方或盲区切入自车轨迹。\n"
    "  4. 全局轨迹冲突：对向越线逆行、路口侧方违规抢行、异物掉落等即将导致碰撞的事件。\n"
    "- 输出“安全”（低风险/系统误触发）：前方及预测轨迹内环境安全，与周围目标相对距离/速度保持安全，"
    "三个视角内均无即将发生碰撞的实体威胁。\n\n"
    "请基于三路全局视野评估客观物理威胁。请仅输出“高风险”或“安全”，不要输出其他任何字符："
)
PROMPT_NO_VIDEO = PROMPT.replace("<video>", "")
N_VIDEO = PROMPT.count("<video>")   # 期望每条样本的视频路数
LABEL_TEXT = {0: "安全", 1: "高风险"}


def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def stem_of(sample):
    videos = sample.get("videos") or []
    return Path(videos[0]).stem if videos else None


def paths_of(sample):
    """ALL video paths of a sample, in list order (front, left, right, ...)."""
    return list(sample.get("videos") or [])


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
                stems.add(Path(line).stem)
    return stems


# ---- output formatters (multi-cam: preserve ALL videos, N_VIDEO placeholders) ----
def to_train(sample):
    paths = paths_of(sample)
    return {
        "messages": [
            {"role": "user", "content": PROMPT},          # N_VIDEO 个 <video>
            {"role": "assistant", "content": LABEL_TEXT[label_of(sample)]},
        ],
        "videos": paths,                                   # 与 <video> 顺序一一对应
    }


def to_test_json(sample):
    # 视频在前(每路一个 item)+ 去掉 <video> 的 prompt 文本
    content = [{"type": "video", "video": p} for p in paths_of(sample)]
    content.append({"type": "text", "text": PROMPT_NO_VIDEO})
    return [{"role": "user", "content": content}]


def to_test_jsonl(sample, max_pixels):
    # 在每个 <video> 处切开：N+1 段文字与 N 个视频交错；丢掉空文字段
    paths = paths_of(sample)
    parts = PROMPT.split("<video>")            # len == N_VIDEO + 1
    content = []
    for i, txt in enumerate(parts):
        if txt:
            content.append({"type": "text", "text": txt})
        if i < len(paths):
            content.append({"type": "video", "video": paths[i], "max_pixels": max_pixels})
    return {"id": stem_of(sample), "label": label_of(sample), "content": content}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corrected_json", required=True,
                    help="Label-corrected full dataset (assistant content 0/1 or 安全/高风险)")
    ap.add_argument("--drop_stems_file", default=None,
                    help="Newline list of stems/paths to exclude (e.g. low-frame videos)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--train_pos", type=int, required=True, help="# 高风险(1) in train")
    ap.add_argument("--train_neg", type=int, required=True, help="# 安全(0) in train")
    ap.add_argument("--test_pos", type=int, required=True, help="# 高风险(1) in test")
    ap.add_argument("--test_neg", type=int, required=True, help="# 安全(0) in test")
    ap.add_argument("--true_hard_json", default=None,
                    help="Optional genuine-hard-examples json to oversample in TRAIN")
    ap.add_argument("--oversample_factor", type=int, default=1,
                    help="Train-side hard examples appear this many times (default 1 = none)")
    ap.add_argument("--max_pixels", type=int, default=336000,
                    help="max_pixels for the video segment in test.jsonl (default 336000)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    samples = load_json(args.corrected_json)

    # ---- multi-cam sanity: every sample must have exactly N_VIDEO videos ----
    vid_counts = Counter(len(s.get("videos") or []) for s in samples)
    print(f"[multicam] PROMPT 期望每条 {N_VIDEO} 路视频；样本视频路数分布: {dict(vid_counts)}")
    if samples:
        print(f"[multicam] sample[0].videos（核对顺序应为 前→左→右）:")
        for p in (samples[0].get("videos") or []):
            print(f"             {p}")
    bad = [s for s in samples if len(s.get("videos") or []) != N_VIDEO]
    if bad:
        raise SystemExit(
            f"❌ {len(bad)} 条样本的视频路数 ≠ {N_VIDEO}（PROMPT 的 <video> 个数）。"
            f"\n   若数据是单视角，改回单视角 PROMPT；若是 3cam，检查数据管道是否每条都出齐 3 路。")

    drop_stems = load_drop_stems(args.drop_stems_file)
    hard_stems = set()
    if args.true_hard_json:
        hard_stems = {stem_of(s) for s in load_json(args.true_hard_json)}
        hard_stems.discard(None)

    kept = [s for s in samples if stem_of(s) not in drop_stems]
    n_dropped = len(samples) - len(kept)

    pos = [s for s in kept if label_of(s) == 1]
    neg = [s for s in kept if label_of(s) == 0]
    random.shuffle(pos)
    random.shuffle(neg)

    need_pos = args.test_pos + args.train_pos
    need_neg = args.test_neg + args.train_neg
    if need_pos > len(pos):
        raise SystemExit(f"❌ 高风险不足: 需要 {need_pos} (test {args.test_pos}+train {args.train_pos}) "
                         f"但只有 {len(pos)}")
    if need_neg > len(neg):
        raise SystemExit(f"❌ 安全不足: 需要 {need_neg} (test {args.test_neg}+train {args.train_neg}) "
                         f"但只有 {len(neg)}")

    # reserve test first, then train from the remainder (guarantees disjoint)
    test_samples = pos[:args.test_pos] + neg[:args.test_neg]
    train_samples = (pos[args.test_pos:args.test_pos + args.train_pos]
                     + neg[args.test_neg:args.test_neg + args.train_neg])
    random.shuffle(test_samples)
    random.shuffle(train_samples)

    # leakage guard
    train_stems = {stem_of(s) for s in train_samples}
    test_stems = {stem_of(s) for s in test_samples}
    overlap = train_stems & test_stems
    assert not overlap, f"LEAKAGE: {len(overlap)} stems in both train and test!"

    # optional oversample (train only)
    hard_in_train = [s for s in train_samples if stem_of(s) in hard_stems]
    extra = hard_in_train * max(0, args.oversample_factor - 1)
    train_final_samples = train_samples + extra
    random.shuffle(train_final_samples)

    # ---- write ----
    train_out = [to_train(s) for s in train_final_samples]
    test_json_out = [to_test_json(s) for s in test_samples]
    with open(out / "train_final.json", "w", encoding="utf-8") as f:
        json.dump(train_out, f, ensure_ascii=False, indent=2)
    with open(out / "test.json", "w", encoding="utf-8") as f:
        json.dump(test_json_out, f, ensure_ascii=False, indent=2)
    with open(out / "test.jsonl", "w", encoding="utf-8") as f:
        for s in test_samples:
            f.write(json.dumps(to_test_jsonl(s, args.max_pixels), ensure_ascii=False) + "\n")

    # report
    print("\n" + "=" * 72)
    print("📦 Final dataset assembled (explicit count control)")
    print("=" * 72)
    print(f"  pool: {len(samples)}  dropped {n_dropped}  → kept {len(kept)} "
          f"(高风险 {len(pos)}, 安全 {len(neg)})")
    print(f"  ── test ──")
    print(f"    高风险 {args.test_pos} + 安全 {args.test_neg} = {len(test_samples)}")
    print(f"  ── train (base) ──")
    print(f"    高风险 {args.train_pos} + 安全 {args.train_neg} = {len(train_samples)}")
    if args.oversample_factor > 1:
        print(f"  ── oversample (factor={args.oversample_factor}, train only) ──")
        print(f"    hard in train: {len(hard_in_train)} → +{len(extra)} copies")
    print(f"    train_final: {len(train_final_samples)}")
    print(f"\n  ✅ leakage check passed: train ∩ test = ∅")
    print(f"  ↳ {out / 'train_final.json'}  ({len(train_out)})")
    print(f"  ↳ {out / 'test.json'}        ({len(test_json_out)})")
    print(f"  ↳ {out / 'test.jsonl'}       ({len(test_samples)})")
    print("=" * 72)


if __name__ == "__main__":
    main()
