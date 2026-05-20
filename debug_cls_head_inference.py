"""Debug script for cls_head inference — verifies every step on a single sample.

Usage:
    python debug_cls_head_inference.py \
        --merged_dir /path/to/merged_model \
        --video /path/to/test_video.mp4 \
        --prompt-file /path/to/prompt.txt   # optional, uses default if missing
        --device npu:0

Outputs detailed diagnostics:
1. cls_head weight statistics (verify load worked)
2. Tokenized prompt (verify chat template is correct)
3. Hidden states at multiple positions (verify position offset)
4. cls_head outputs at multiple positions
5. Comparison of two chat template paths
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401

# Path to LlamaFactory source
sys.path.insert(0, "/home/ma-user/work/lyf/LlamaFactory-qwen35/src")
from llamafactory.model.cls_head import BinaryClassificationHead  # noqa: E402

try:
    from transformers import Qwen3_5VLForConditionalGeneration as TargetVLModel, AutoProcessor, AutoTokenizer
except ImportError:
    from transformers import AutoModelForImageTextToText as TargetVLModel, AutoProcessor, AutoTokenizer

from qwen_vl_utils import process_vision_info


# IMPORTANT: this MUST match the prompt used during training exactly.
# Any difference (even punctuation / whitespace) changes the tokenization
# and shifts the hidden state at the last position — cls_head was trained
# on a specific distribution of "last hidden states", so OOD prompts cause
# uniformly biased outputs.
DEFAULT_PROMPT = (
    "你是一个自动驾驶安全专家。请观看以下车辆行驶视频：自车前视视角<video>\n"
    "任务：自动驾驶前视场景碰撞风险二分类。\n"
    "请严格根据物理环境和车辆动态，判断当前自车是否面临真实的碰撞风险。\n\n"
    "【判定规则】：\n"
    "- 输出“高风险”（真实危险）：自车行驶轨迹上存在即将发生物理碰撞的实体威胁。"
    "只要客观环境构成了紧急碰撞危险，均属于此类，包括但不限于以下典型场景：\n"
    "  1. 绝对距离压迫：正前方已有明确的实体障碍物（或静止目标）极度逼近。\n"
    "  2. 纵向追尾/相对速度危险：自车车速过快或前方目标骤停，导致两者的相对距离在画面中急速缩短。\n"
    "  3. 横向/盲区突发侵入：视野盲区或道路两侧突然有目标（行人、两轮车、其他车辆等）横向切入自车轨迹。\n"
    "  4. 全局轨迹冲突：如对向车辆失控越线逆行、路口侧方车辆违规抢行、异物掉落等任何即将导致真实碰撞的危险事件。\n"
    "- 输出“安全”（低风险/系统误触发）：前方及预测轨迹内环境安全，"
    "与周围目标的相对距离/速度保持安全，无任何即将发生碰撞的实体威胁。"
    "**特别注意：即使视频画面出现剧烈抖动、车头明显下沉（表示自车正在急刹减速），"
    "只要客观上并没有真正会撞上的实体障碍物，均属于系统误触发，必须严格输出“安全”。**\n\n"
    "请忽略自车不必要的减速动作，基于全局视野评估客观物理威胁。"
    "请仅输出“高风险”或“安全”，不要输出其他任何字符："
)


def print_section(title):
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print("=" * 80)


def stats(t, name):
    t = t.float() if t.is_floating_point() else t
    return (
        f"{name}: shape={tuple(t.shape)} "
        f"mean={t.mean().item():.4f} std={t.std().item():.4f} "
        f"min={t.min().item():.4f} max={t.max().item():.4f}"
    )


def run_batch(args, model, processor, head, meta):
    """Batch mode: read JSONL, run cls_head on each, print summary."""
    label_map = meta["label_map"]
    pos_text = label_map["1"]
    neg_text = label_map["0"]
    head_dtype = next(head.parameters()).dtype

    # Read manifest. Accepts two formats:
    #   (A) JSONL: each line a dict {"video": path, "label": "高风险"|"安全"}  (label optional)
    #   (B) JSON list (legacy infer format): [[{"role":"user","content":[{"type":"video","video":...},{"type":"text","text":...}]}], ...]
    samples = []
    with open(args.manifest, encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            # legacy JSON list
            data = json.load(f)
            for entry in data:
                msg = entry[0] if isinstance(entry, list) else entry
                content = msg.get("content", [])
                video = next((c["video"] for c in content if c.get("type") == "video"), None)
                text = next((c["text"] for c in content if c.get("type") == "text"), args.prompt)
                samples.append({"video": video, "prompt": text, "label": None})
        else:
            # JSONL
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                samples.append({
                    "video": row.get("video") or row.get("video_path") or row.get("id"),
                    "prompt": row.get("prompt", args.prompt),
                    "label": row.get("label"),
                })

    print(f"\nLoaded {len(samples)} samples from {args.manifest}")
    print(f"{'='*100}")
    print(f"{'idx':>4} | {'label':>8} | {'p_safe':>8} | {'p_risk':>8} | {'pred':>8} | {'OK':>3} | video")
    print(f"{'-'*100}")

    results = []
    for i, s in enumerate(samples):
        try:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "video", "video": s["video"], "fps": args.video_fps, "max_pixels": args.video_max_pixels},
                    {"type": "text", "text": s["prompt"]},
                ],
            }]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            text = text.replace("<think>\n\n</think>\n\n", "")
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages, return_video_kwargs=True, image_patch_size=16, return_video_metadata=True,
            )
            video_metadatas = None
            if video_inputs and len(video_inputs) > 0 and isinstance(video_inputs[0], tuple):
                video_inputs, video_metadatas = zip(*video_inputs)
                video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs,
                video_metadata=video_metadatas, padding=True, return_tensors="pt", **video_kwargs,
            ).to(args.device)

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
            h = outputs.hidden_states[-1][:, -1, :]
            cls_logits = head(h.to(head_dtype))
            probs = torch.softmax(cls_logits.float(), dim=-1)
            p_neg, p_pos = probs[0, 0].item(), probs[0, 1].item()
            pred = pos_text if p_pos >= 0.5 else neg_text

            correct = ""
            if s["label"] is not None:
                correct = "✓" if pred == s["label"] else "✗"

            results.append({
                "video": s["video"], "label": s["label"],
                "p_neg": p_neg, "p_pos": p_pos, "pred": pred, "correct": correct,
            })
            vid_short = (s["video"] or "").split("/")[-1][:40]
            print(f"{i:>4} | {(s['label'] or '?'):>8} | {p_neg:>8.4f} | {p_pos:>8.4f} | {pred:>8} | {correct:>3} | {vid_short}")
        except Exception as e:
            print(f"{i:>4} | ERROR: {e}")
            results.append({"video": s["video"], "label": s["label"], "error": str(e)})

    # Summary
    print(f"{'='*100}")
    valid = [r for r in results if "error" not in r]
    if not valid:
        print("No valid predictions.")
        return

    has_labels = all(r["label"] is not None for r in valid)
    if has_labels:
        pos_samples = [r for r in valid if r["label"] == pos_text]
        neg_samples = [r for r in valid if r["label"] == neg_text]
        correct_count = sum(1 for r in valid if r["correct"] == "✓")
        print(f"\nSummary (n={len(valid)}):")
        print(f"  Accuracy: {correct_count}/{len(valid)} = {100*correct_count/len(valid):.2f}%")
        if pos_samples:
            ps = [r["p_pos"] for r in pos_samples]
            print(f"  '{pos_text}' samples (n={len(pos_samples)}): "
                  f"P(pos) min={min(ps):.3f} mean={sum(ps)/len(ps):.3f} max={max(ps):.3f}")
        if neg_samples:
            ns = [r["p_pos"] for r in neg_samples]
            print(f"  '{neg_text}' samples (n={len(neg_samples)}): "
                  f"P(pos) min={min(ns):.3f} mean={sum(ns)/len(ns):.3f} max={max(ns):.3f}")
        # Confusion matrix at threshold 0.5
        tp = sum(1 for r in pos_samples if r["p_pos"] >= 0.5)
        fn = len(pos_samples) - tp
        tn = sum(1 for r in neg_samples if r["p_pos"] < 0.5)
        fp = len(neg_samples) - tn
        print(f"\n  Confusion @ thresh=0.5:")
        print(f"    TP={tp}  FN={fn}  TN={tn}  FP={fp}")
        if tp + fp > 0:
            print(f"    Precision (pos): {tp/(tp+fp):.4f}")
        if tp + fn > 0:
            print(f"    Recall (pos):    {tp/(tp+fn):.4f}")
    else:
        ps = [r["p_pos"] for r in valid]
        print(f"\nNo labels provided. P(pos) distribution (n={len(valid)}):")
        print(f"  min={min(ps):.3f}  max={max(ps):.3f}  mean={sum(ps)/len(ps):.3f}")
        n_pos = sum(1 for p in ps if p >= 0.5)
        print(f"  Predicted '{pos_text}': {n_pos}/{len(valid)} ({100*n_pos/len(valid):.1f}%)")
        print(f"  Predicted '{neg_text}': {len(valid)-n_pos}/{len(valid)} ({100*(len(valid)-n_pos)/len(valid):.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged_dir", required=True)
    parser.add_argument("--video", default=None, help="Single video for deep diagnostic (mutually exclusive with --manifest)")
    parser.add_argument("--manifest", default=None, help="JSONL/JSON manifest for batch mode")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--video_fps", type=float, default=8.0)
    parser.add_argument("--video_max_pixels", type=int, default=589824)
    args = parser.parse_args()

    if (args.video is None) == (args.manifest is None):
        raise SystemExit("Specify exactly one of --video (single deep diagnostic) or --manifest (batch).")

    device = args.device
    torch.npu.set_device(device)

    # =================================================================
    print_section("STEP 1: Load cls_head and verify weights")
    # =================================================================
    head_bin = Path(args.merged_dir) / "cls_head.bin"
    head_meta = Path(args.merged_dir) / "cls_head_meta.json"
    print(f"cls_head.bin path: {head_bin}")
    print(f"cls_head_meta.json path: {head_meta}")
    print(f"cls_head.bin exists: {head_bin.exists()}")
    print(f"cls_head.bin size: {head_bin.stat().st_size} bytes")

    with open(head_meta, encoding="utf-8") as f:
        meta = json.load(f)
    print(f"\nMeta content:\n{json.dumps(meta, ensure_ascii=False, indent=2)}")

    # Tokenization sanity check: cls_head/training only use the FIRST token id
    # of each class text. If a class name tokenizes to multiple tokens, the
    # FIRST token must still be unique to that class for labels to be correct.
    # For Qwen3.5 with 高风险/安全, both are single tokens — but new tasks may
    # use longer phrases that tokenize differently.
    from transformers import AutoTokenizer as _Tok  # noqa: F401

    # Create head and check FRESHLY INITIALIZED weights
    fresh_head = BinaryClassificationHead(meta["hidden_size"], dropout=meta.get("dropout", 0.0))
    print(f"\n--- FRESH (untrained) head weight stats ---")
    print(stats(fresh_head.dense.weight, "dense.weight"))
    print(stats(fresh_head.out_proj.weight, "out_proj.weight"))
    print(f"out_proj.bias = {fresh_head.out_proj.bias.tolist()}")

    # Load trained weights
    loaded_state = torch.load(head_bin, map_location="cpu")
    print(f"\n--- Loaded state_dict keys ---")
    for k, v in loaded_state.items():
        print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")

    missing, unexpected = fresh_head.load_state_dict(loaded_state, strict=False)
    print(f"\nLoad result: missing={missing}, unexpected={unexpected}")

    print(f"\n--- TRAINED head weight stats (after load) ---")
    print(stats(fresh_head.dense.weight, "dense.weight"))
    print(stats(fresh_head.out_proj.weight, "out_proj.weight"))
    print(f"out_proj.bias = {fresh_head.out_proj.bias.tolist()}")
    print(f"\nout_proj.weight[0] (safe class, first 10): {fresh_head.out_proj.weight[0, :10].tolist()}")
    print(f"out_proj.weight[1] (risk class, first 10): {fresh_head.out_proj.weight[1, :10].tolist()}")

    # CHECK: best signal is out_proj.bias — fresh init is exactly 0, training pushes it away.
    # The std-based check was too aggressive: for cls_head fine-tune with lr=1e-5, std only
    # grows ~2-5% over full training, so std stays at ~0.020 even when fully trained.
    bias_abs_max = fresh_head.out_proj.bias.abs().max().item()
    print(f"\nout_proj.bias |max| = {bias_abs_max:.2e}  (fresh init is exactly 0.0)")
    if bias_abs_max < 1e-8:
        print("⚠️  WARNING: out_proj.bias is essentially zero → cls_head was NOT trained at all.")
    elif bias_abs_max < 1e-6:
        print("⚠️  Suspicious: out_proj.bias moved very little. Either very early in training,")
        print("    or training is broken (e.g. bf16 underflow).")
    else:
        print("✅ Weights show evidence of training (out_proj.bias is non-trivially non-zero).")

    head = fresh_head.to(device).to(torch.bfloat16).eval()

    # =================================================================
    print_section("STEP 2: Load model + processor")
    # =================================================================
    print(f"Loading model from {args.merged_dir}")
    model = TargetVLModel.from_pretrained(
        args.merged_dir,
        torch_dtype="auto",
        device_map=None,
        attn_implementation=os.getenv("INFER_ATTN_IMPL", "sdpa"),
    ).eval().to(device)
    processor = AutoProcessor.from_pretrained(args.merged_dir)
    tokenizer = processor.tokenizer
    print(f"Model dtype: {next(model.parameters()).dtype}")
    print(f"Tokenizer padding_side: {tokenizer.padding_side}")

    # Class-text tokenization sanity check
    pos_text = meta.get("positive_text", meta.get("label_map", {}).get("1", "高风险"))
    neg_text = meta.get("negative_text", meta.get("label_map", {}).get("0", "安全"))
    pos_ids = tokenizer.encode(pos_text, add_special_tokens=False)
    neg_ids = tokenizer.encode(neg_text, add_special_tokens=False)
    print(f"\nClass tokenization:")
    print(f"  positive '{pos_text}' -> {pos_ids}  ({len(pos_ids)} token{'s' if len(pos_ids) > 1 else ''})")
    print(f"  negative '{neg_text}' -> {neg_ids}  ({len(neg_ids)} token{'s' if len(neg_ids) > 1 else ''})")
    if len(pos_ids) > 1 or len(neg_ids) > 1:
        print(f"  ⚠️  At least one class is multi-token. The code uses only the FIRST token id"
              f" for labeling; that's fine as long as the first tokens differ ({pos_ids[0]} vs {neg_ids[0]})"
              f" — but be aware that for the token-CE inference path, the first-step logit"
              f" you read corresponds to the FIRST sub-token, not the full phrase.")
    else:
        print(f"  ✅ Both classes are single tokens — no ambiguity.")
    assert pos_ids[0] != neg_ids[0], "FIRST tokens of two classes must differ!"

    # =================================================================
    # Batch branch: if --manifest given, run batch eval and exit early.
    # =================================================================
    if args.manifest is not None:
        print_section("BATCH MODE: running cls_head on manifest")
        run_batch(args, model, processor, head, meta)
        return

    # =================================================================
    print_section("STEP 3: Build prompt and inspect tokenization")
    # =================================================================
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": args.video, "fps": args.video_fps, "max_pixels": args.video_max_pixels},
            {"type": "text", "text": args.prompt},
        ],
    }]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    print(f"--- Chat template output (raw text, BEFORE strip) ---")
    print(repr(text[-200:]))
    # Strip empty think block (Qwen3.5 chat template inserts it even with enable_thinking=False,
    # but the training template qwen3_5_nothink does NOT have it).
    text = text.replace("<think>\n\n</think>\n\n", "")
    print(f"--- AFTER stripping <think></think> ---")
    print(repr(text[-200:]))
    print(f"\nText ends with: {repr(text[-50:])}")

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        return_video_kwargs=True,
        image_patch_size=16,
        return_video_metadata=True,
    )

    video_metadatas = None
    if video_inputs is not None and len(video_inputs) > 0:
        if isinstance(video_inputs, (tuple, list)):
            video_inputs, video_metadatas = zip(*video_inputs)
            video_inputs = list(video_inputs)
            video_metadatas = list(video_metadatas)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        video_metadata=video_metadatas,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    ).to(device)

    input_ids = inputs.input_ids[0]  # [T]
    print(f"\nTotal tokens: {input_ids.size(0)}")
    print(f"Last 30 token IDs: {input_ids[-30:].tolist()}")
    print(f"Last 30 tokens decoded individually:")
    for i, tid in enumerate(input_ids[-30:].tolist()):
        decoded = tokenizer.decode([tid])
        print(f"  pos[-{30 - i}] id={tid} = {repr(decoded)}")
    print(f"\nLast token: id={input_ids[-1].item()}, decoded={repr(tokenizer.decode([input_ids[-1].item()]))}")
    print(f"Last 3 tokens joined: {repr(tokenizer.decode(input_ids[-3:].tolist()))}")

    # =================================================================
    print_section("STEP 4: Forward pass, extract hidden states")
    # =================================================================
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)

    last_hidden = outputs.hidden_states[-1]  # [1, T, D]
    print(f"\nlast_hidden shape: {last_hidden.shape}")
    print(stats(last_hidden, "last_hidden (all positions)"))

    # Try multiple positions
    print(f"\n--- Hidden state at last 6 positions ---")
    for off in range(-6, 0):
        h_pos = last_hidden[:, off, :]
        tok_id = input_ids[off].item()
        tok_str = tokenizer.decode([tok_id])
        print(f"  pos[{off}] (token={repr(tok_str)}): "
              f"mean={h_pos.float().mean().item():.4f} std={h_pos.float().std().item():.4f}")

    # =================================================================
    print_section("STEP 5: cls_head outputs at multiple positions")
    # =================================================================
    print("\n--- cls_head output at last 6 positions ---")
    print(f"{'pos':>5} | {'token':>20} | {'logit_safe':>11} | {'logit_risk':>11} | {'P(risk)':>8}")
    print("-" * 70)
    for off in range(-6, 0):
        h_pos = last_hidden[:, off, :]
        cls_logits = head(h_pos.to(torch.bfloat16))  # [1, 2]
        probs = torch.softmax(cls_logits.float(), dim=-1)
        tok_id = input_ids[off].item()
        tok_str = tokenizer.decode([tok_id])
        print(f"  {off:>3} | {repr(tok_str)[:20]:>20} | "
              f"{cls_logits[0, 0].item():>11.4f} | "
              f"{cls_logits[0, 1].item():>11.4f} | "
              f"{probs[0, 1].item():>7.4f}")

    # =================================================================
    print_section("STEP 6: Try the training-style path (manual answer position)")
    # =================================================================
    # In training, the input had the answer appended: prompt + "高风险" or "安全"
    # The hidden_pos was answer_pos - 1, i.e., the position right BEFORE the first answer token.
    # In inference with add_generation_prompt=True, this should be the LAST position.
    # BUT — let's verify by appending a dummy answer and finding the position
    text_with_answer = text + "高风险<|im_end|>"
    print(f"Text with appended answer ends with: {repr(text_with_answer[-50:])}")
    inputs_with_answer = processor(
        text=[text_with_answer],
        images=image_inputs,
        videos=video_inputs,
        video_metadata=video_metadatas,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    ).to(device)
    ids_with_ans = inputs_with_answer.input_ids[0]
    print(f"Length with answer: {ids_with_ans.size(0)} (inference-only was {input_ids.size(0)})")

    # Find first occurrence of high_risk token (id from meta)
    high_risk_id = meta["positive_token_id"]
    safe_id = meta["negative_token_id"]
    print(f"Looking for positive_token_id={high_risk_id} ('高风险') or negative_token_id={safe_id} ('安全')")

    pos_positions = (ids_with_ans == high_risk_id).nonzero(as_tuple=True)[0]
    neg_positions = (ids_with_ans == safe_id).nonzero(as_tuple=True)[0]
    print(f"high_risk_id positions in ids_with_answer: {pos_positions.tolist()}")
    print(f"safe_id positions in ids_with_answer: {neg_positions.tolist()}")

    if pos_positions.numel() > 0:
        # IMPORTANT: 高风险/安全 may appear multiple times in the prompt itself
        # (the rules text mentions both classes). The actual appended answer is
        # always the LAST occurrence. Take that, not the first.
        answer_pos = pos_positions[-1].item()
        hidden_pos_training = answer_pos - 1
        print(f"  (high_risk_id appears at: {pos_positions.tolist()})")
        print(f"  taking LAST occurrence (the appended answer): answer_pos={answer_pos}")
        print(f"Training-style answer_pos = {answer_pos}, hidden_pos = {hidden_pos_training}")
        # Run forward with answer included
        with torch.no_grad():
            outputs_w_ans = model(**inputs_with_answer, output_hidden_states=True, return_dict=True, use_cache=False)
        last_hidden_w_ans = outputs_w_ans.hidden_states[-1]
        h_train_style = last_hidden_w_ans[:, hidden_pos_training, :]
        cls_logits_train = head(h_train_style.to(torch.bfloat16))
        probs_train = torch.softmax(cls_logits_train.float(), dim=-1)
        print(f"\n--- Training-style hidden state (pos {hidden_pos_training} of prompt-with-answer) ---")
        print(f"  logit_safe={cls_logits_train[0, 0].item():.4f}, "
              f"logit_risk={cls_logits_train[0, 1].item():.4f}, "
              f"P(risk)={probs_train[0, 1].item():.4f}")

        # Compare to inference-only path
        h_inference = last_hidden[:, -1, :]
        diff = (h_train_style - h_inference).float().abs().mean().item()
        cos = torch.nn.functional.cosine_similarity(h_train_style.float(), h_inference.float()).item()
        print(f"\n--- Compare training-style vs inference-only hidden state ---")
        print(f"  L1 diff per dim: {diff:.6f}")
        print(f"  Cosine similarity: {cos:.6f}")
        print(f"  (If cos < 0.99, the two paths differ significantly — that's the bug)")

    # =================================================================
    print_section("STEP 7: Visual encoder sanity check")
    # =================================================================
    # If hidden states at the last position are dominated by visual features (which they should be),
    # check visual token count
    print(f"\ninputs.pixel_values_videos shape: {getattr(inputs, 'pixel_values_videos', None)}")
    if hasattr(inputs, "pixel_values_videos") and inputs.pixel_values_videos is not None:
        print(f"  pixel_values_videos: shape={tuple(inputs.pixel_values_videos.shape)}, "
              f"dtype={inputs.pixel_values_videos.dtype}")
    if hasattr(inputs, "video_grid_thw") and inputs.video_grid_thw is not None:
        print(f"  video_grid_thw: {inputs.video_grid_thw.tolist()}")
        t, h, w = inputs.video_grid_thw[0].tolist()
        total_patches_per_frame = h * w
        total_frames = t * 2  # temporal_patch_size=2
        print(f"  → temporal blocks={t}, spatial grid={h}x{w}={total_patches_per_frame}")
        print(f"  → ~{total_frames} input frames, ~{t * total_patches_per_frame // 4} visual tokens after spatial_merge=2")
        if total_patches_per_frame > 2304:
            print(f"  ⚠️  WARNING: spatial patches {total_patches_per_frame} > pos_embed table size 2304!")

    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
