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


DEFAULT_PROMPT = """你是一个自动驾驶安全专家。请观看以下车辆行驶视频：自车前视视角<video>
任务：自动驾驶前视场景碰撞风险二分类。
请严格根据物理环境和车辆动态，判断当前自车是否面临真实的碰撞风险。
请仅输出"高风险"或"安全"，不要输出其他任何字符："""


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged_dir", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--video_fps", type=float, default=8.0)
    parser.add_argument("--video_max_pixels", type=int, default=589824)
    args = parser.parse_args()

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

    # CHECK: if dense.weight std is ~0.02 (init value), weights weren't actually trained
    if fresh_head.dense.weight.std().item() < 0.025:
        print("⚠️  WARNING: dense.weight.std is very close to init value (~0.02). "
              "This suggests cls_head was NOT actually trained!")
    else:
        print("✅ Weight stats look trained (dense.weight.std > 0.025)")

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
    print(f"--- Chat template output (raw text) ---")
    print(repr(text[-500:]))  # last 500 chars
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
        answer_pos = pos_positions[0].item()
        hidden_pos_training = answer_pos - 1
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
