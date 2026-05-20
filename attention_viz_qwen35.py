"""Attention visualization for Qwen3.5-VL on a single video.

Outputs a grid of (original frame) + (attention heatmap) + (overlay) per temporal
chunk, showing which spatio-temporal regions the model attended to when producing
the answer token.

Usage:
    python attention_viz_qwen35.py \
        --merged_dir /home/ma-user/work/lyf/outmodel/<merged> \
        --video /path/to/test_video.mp4 \
        --output_dir /home/ma-user/work/lyf/viz_out/sample01 \
        --video_max_pixels 589824 \
        --video_fps 8.0

Notes:
- Uses attn_implementation="eager" so model returns attention weights.
  This is SLOWER than sdpa/fa2 — use only for debug, not production.
- Qwen3.5 is hybrid (3 linear + 1 full attention per group). Only the 8
  full_attention layers (idx 3,7,11,15,19,23,27,31) return meaningful weights;
  linear_attention layers return None or zeros — those are skipped.
- Memory: full attention matrix is large. We immediately slice to keep only the
  last query position's attention, avoiding O(T^2) storage.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch_npu  # noqa: F401
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

try:
    from transformers import Qwen3_5VLForConditionalGeneration as TargetVLModel, AutoProcessor
except ImportError:
    from transformers import AutoModelForImageTextToText as TargetVLModel, AutoProcessor

from qwen_vl_utils import process_vision_info


DEFAULT_PROMPT = (
    "你是一个自动驾驶安全专家。请观看以下车辆行驶视频：自车前视视角<video>\n"
    "任务：自动驾驶前视场景碰撞风险二分类。\n"
    "请严格根据物理环境和车辆动态，判断当前自车是否面临真实的碰撞风险。\n"
    "请仅输出「高风险」或「安全」，不要输出其他任何字符："
)


# ============================================================
# Step 1: load model + processor with eager attention
# ============================================================
def load_model(merged_dir, device):
    print(f"Loading model from {merged_dir}  (attn=eager for visualization)")
    model = TargetVLModel.from_pretrained(
        merged_dir,
        torch_dtype="auto",
        device_map=None,
        attn_implementation="eager",   # CRITICAL: required to get attention weights
    ).eval().to(device)
    processor = AutoProcessor.from_pretrained(merged_dir)
    return model, processor


# ============================================================
# Step 2: build inputs + find visual token range
# ============================================================
def build_inputs(processor, video_path, prompt, video_fps, video_max_pixels, device):
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": video_path, "fps": video_fps, "max_pixels": video_max_pixels},
            {"type": "text", "text": prompt},
        ],
    }]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    # strip empty think block to match training prompt
    text = text.replace("<think>\n\n</think>\n\n", "")

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True, image_patch_size=16, return_video_metadata=True,
    )
    video_metadatas = None
    if video_inputs and isinstance(video_inputs[0], tuple):
        video_inputs, video_metadatas = zip(*video_inputs)
        video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)

    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        video_metadata=video_metadatas, padding=True, return_tensors="pt", **video_kwargs,
    ).to(device)
    return inputs, video_inputs


def find_visual_token_range(input_ids, vision_start_id, vision_end_id):
    """Find [start, end) range of visual tokens in the input sequence."""
    start_positions = (input_ids == vision_start_id).nonzero(as_tuple=True)[0]
    end_positions = (input_ids == vision_end_id).nonzero(as_tuple=True)[0]
    if len(start_positions) == 0 or len(end_positions) == 0:
        raise RuntimeError(
            f"Could not find vision tokens. start_id={vision_start_id}, end_id={vision_end_id}. "
            f"Available special tokens in input: {input_ids[input_ids >= 248000].tolist()[:20]}"
        )
    # visual content sits BETWEEN <|vision_start|> and <|vision_end|>
    return int(start_positions[0].item()) + 1, int(end_positions[0].item())


# ============================================================
# Step 3: forward + extract attention from last token to visual tokens
# ============================================================
@torch.no_grad()
def extract_visual_attention(model, inputs, visual_token_start, visual_token_end, last_n_layers=4):
    """Run forward with output_attentions=True, return mean attention from the
    LAST input position to each visual token, aggregated over the last N
    full_attention layers and over heads.

    Returns: tensor of shape [num_visual_tokens]
    """
    outputs = model(
        **inputs,
        output_attentions=True,
        return_dict=True,
        use_cache=False,
    )
    # outputs.attentions is a tuple of length num_layers
    # each element is None (for linear_attention) or [B, H, T, T] (for full_attention)
    full_attn_layers = []
    for i, attn in enumerate(outputs.attentions):
        if attn is not None and attn.dim() == 4:
            full_attn_layers.append((i, attn))
    print(f"  Got {len(full_attn_layers)} full-attention layers out of {len(outputs.attentions)}")

    if not full_attn_layers:
        raise RuntimeError("No usable attention weights returned. Model may be using SDPA/FA2 despite eager request.")

    # Take last N full-attention layers (deeper layers carry more semantic info)
    selected = full_attn_layers[-last_n_layers:]
    print(f"  Aggregating over layers: {[i for i, _ in selected]}")

    # For each, extract attention from LAST query position to visual tokens
    per_layer_attn = []
    last_q_pos = -1  # last input token's attention
    for layer_idx, attn in selected:
        # attn shape: [B=1, num_heads, T, T]
        # [0, :, last_q_pos, visual_token_start:visual_token_end]
        a = attn[0, :, last_q_pos, visual_token_start:visual_token_end]  # [num_heads, num_visual]
        a_mean_heads = a.float().mean(dim=0)  # [num_visual]
        per_layer_attn.append(a_mean_heads)
        del attn  # free memory

    # Average over selected layers
    attention_scores = torch.stack(per_layer_attn, dim=0).mean(dim=0)  # [num_visual]
    return attention_scores


# ============================================================
# Step 4: reshape visual attention back to (T_chunks, H_merged, W_merged)
# ============================================================
def reshape_attention_to_grid(attention_scores, video_grid_thw, spatial_merge_size=2):
    """video_grid_thw is the ViT's patch grid before spatial_merge.
    After spatial_merge_size=2, LLM sees (T, H/2, W/2) visual tokens.
    """
    T, H, W = video_grid_thw.tolist()
    H_merged = H // spatial_merge_size
    W_merged = W // spatial_merge_size
    num_visual = T * H_merged * W_merged
    if attention_scores.numel() != num_visual:
        raise RuntimeError(
            f"Visual token count mismatch: scores={attention_scores.numel()} vs "
            f"expected T*H_m*W_m={T}*{H_merged}*{W_merged}={num_visual}. "
            f"Original grid_thw=({T},{H},{W}), merge_size={spatial_merge_size}."
        )
    return attention_scores.reshape(T, H_merged, W_merged).cpu().numpy()


# ============================================================
# Step 5: extract video frames for visualization
# ============================================================
def extract_frames_from_video_input(video_inputs):
    """video_inputs from qwen_vl_utils.process_vision_info is a list of tensor
    [T*2, C, H, W] per video (already resampled). Convert to list of PIL frames.
    """
    vid = video_inputs[0]  # first (and only) video
    if isinstance(vid, list):
        # might be a list of PIL frames already
        return vid
    # tensor [T_frames, 3, H, W] in [0, 1] or [0, 255]
    if isinstance(vid, torch.Tensor):
        vid = vid.cpu().numpy()
    # vid shape varies; standardize to (N, H, W, 3)
    if vid.ndim == 4 and vid.shape[1] == 3:
        vid = vid.transpose(0, 2, 3, 1)
    if vid.dtype != np.uint8:
        # assume 0-1 or -1-1
        if vid.min() < -0.1:
            vid = (vid + 1) * 127.5
        elif vid.max() <= 1.5:
            vid = vid * 255
        vid = np.clip(vid, 0, 255).astype(np.uint8)
    return [Image.fromarray(vid[i]) for i in range(vid.shape[0])]


# ============================================================
# Step 6: render visualization grid
# ============================================================
def render_visualization(attention_grid, frames, output_dir, p_safe, p_risk, video_path):
    """attention_grid: (T_chunks, H_merged, W_merged) numpy array.
    frames: list of PIL Images (length should be 2*T_chunks since temporal_patch_size=2).
    """
    T = attention_grid.shape[0]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Normalize attention to [0, 1] across all chunks for consistent colormap
    a_min, a_max = attention_grid.min(), attention_grid.max()
    norm_attention = (attention_grid - a_min) / (a_max - a_min + 1e-8)

    # For each temporal chunk, render: original frame | heatmap | overlay
    fig, axes = plt.subplots(T, 3, figsize=(12, 3 * T))
    if T == 1:
        axes = axes[None, :]

    for t in range(T):
        # Pick first frame of this temporal chunk (each chunk = 2 frames merged)
        frame_idx = t * 2 if t * 2 < len(frames) else len(frames) - 1
        frame = frames[frame_idx].convert("RGB")
        frame_arr = np.array(frame)
        H_img, W_img = frame_arr.shape[:2]

        # Upsample heatmap to frame size
        heatmap = norm_attention[t]  # (H_m, W_m)
        heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
        heatmap_resized = heatmap_img.resize((W_img, H_img), Image.BILINEAR)
        heatmap_arr = np.array(heatmap_resized) / 255.0

        # Apply colormap (jet)
        heatmap_rgb = (cm.jet(heatmap_arr)[..., :3] * 255).astype(np.uint8)

        # Overlay (alpha blend)
        alpha = 0.5
        overlay = (frame_arr * (1 - alpha) + heatmap_rgb * alpha).astype(np.uint8)

        # Plot
        axes[t, 0].imshow(frame_arr)
        axes[t, 0].set_title(f"Frame {frame_idx} (chunk {t})")
        axes[t, 0].axis("off")

        axes[t, 1].imshow(heatmap_arr, cmap="jet", vmin=0, vmax=1)
        axes[t, 1].set_title(f"Attention (sum={attention_grid[t].sum():.4f})")
        axes[t, 1].axis("off")

        axes[t, 2].imshow(overlay)
        axes[t, 2].set_title("Overlay")
        axes[t, 2].axis("off")

    fig.suptitle(
        f"{Path(video_path).name}  |  P(safe)={p_safe:.3f}  P(risk)={p_risk:.3f}",
        fontsize=14,
    )
    plt.tight_layout()
    overlay_path = output_dir / "overlay.png"
    plt.savefig(overlay_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved overlay grid to {overlay_path}")

    # Also save attention-per-chunk statistics
    chunk_attention_sum = attention_grid.reshape(T, -1).sum(axis=1)
    stats_path = output_dir / "attention_stats.txt"
    with open(stats_path, "w") as f:
        f.write("Temporal chunk attention distribution\n")
        f.write("=====================================\n")
        f.write(f"Total chunks: {T}\n")
        f.write(f"Attention sum per chunk:\n")
        for t in range(T):
            bar = "█" * int(chunk_attention_sum[t] / chunk_attention_sum.max() * 40)
            f.write(f"  chunk {t:>2}: {chunk_attention_sum[t]:.5f}  {bar}\n")
        f.write(f"\nMost attended chunk: {chunk_attention_sum.argmax()}\n")
        f.write(f"Least attended chunk: {chunk_attention_sum.argmin()}\n")

        # Per-chunk spatial peak location (which (h, w) cell got max attention)
        f.write("\nSpatial peak per chunk (row, col) in merged grid:\n")
        for t in range(T):
            peak_idx = np.unravel_index(np.argmax(attention_grid[t]), attention_grid[t].shape)
            f.write(f"  chunk {t:>2}: peak at ({peak_idx[0]}, {peak_idx[1]}) "
                    f"of grid {attention_grid[t].shape}\n")
    print(f"  Saved stats to {stats_path}")


# ============================================================
# Step 7: get cls predictions for the label in the title
# ============================================================
@torch.no_grad()
def get_predictions(model, processor, inputs):
    """Run a quick forward (no attentions needed this time) to get token logits
    for 安全 and 高风险."""
    # Reuse the outputs from extract_visual_attention if possible — but for
    # clarity we run a separate forward here.
    safe_id = processor.tokenizer.encode("安全", add_special_tokens=False)[0]
    risk_id = processor.tokenizer.encode("高风险", add_special_tokens=False)[0]

    outputs = model(**inputs, return_dict=True, use_cache=False)
    last_logits = outputs.logits[0, -1, :]  # [vocab_size]
    logit_safe = last_logits[safe_id].item()
    logit_risk = last_logits[risk_id].item()
    # softmax over [safe, risk] for P
    e_s = np.exp(logit_safe - max(logit_safe, logit_risk))
    e_r = np.exp(logit_risk - max(logit_safe, logit_risk))
    return e_s / (e_s + e_r), e_r / (e_s + e_r)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged_dir", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--video_fps", type=float, default=8.0)
    parser.add_argument("--video_max_pixels", type=int, default=589824)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--last_n_layers", type=int, default=4,
                        help="How many of the deepest full-attention layers to average over.")
    parser.add_argument("--vision_start_id", type=int, default=248053)
    parser.add_argument("--vision_end_id", type=int, default=248054)
    args = parser.parse_args()

    torch.npu.set_device(args.device)
    os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")

    # 1. Load model
    model, processor = load_model(args.merged_dir, args.device)

    # 2. Build inputs
    print(f"\nProcessing video: {args.video}")
    inputs, video_inputs = build_inputs(
        processor, args.video, args.prompt,
        args.video_fps, args.video_max_pixels, args.device,
    )
    input_ids = inputs.input_ids[0]
    print(f"  Total input tokens: {input_ids.size(0)}")
    print(f"  video_grid_thw: {inputs.video_grid_thw.tolist()}")

    # 3. Find visual token range
    vstart, vend = find_visual_token_range(input_ids, args.vision_start_id, args.vision_end_id)
    print(f"  Visual tokens: positions [{vstart}, {vend}) = {vend - vstart} tokens")

    # 4. Get predictions (forward pass without attention, fast)
    print(f"\nRunning prediction forward...")
    p_safe, p_risk = get_predictions(model, processor, inputs)
    print(f"  P(safe) = {p_safe:.4f}   P(risk) = {p_risk:.4f}")
    pred = "高风险" if p_risk > p_safe else "安全"
    print(f"  Prediction: {pred}")

    # 5. Extract attention
    print(f"\nRunning attention-extraction forward...")
    attention_scores = extract_visual_attention(
        model, inputs, vstart, vend, last_n_layers=args.last_n_layers,
    )
    print(f"  Attention scores shape: {attention_scores.shape}")
    print(f"  Attention range: min={attention_scores.min():.6f}, "
          f"max={attention_scores.max():.6f}, "
          f"mean={attention_scores.mean():.6f}")

    # 6. Reshape to (T, H, W) grid
    attention_grid = reshape_attention_to_grid(attention_scores, inputs.video_grid_thw[0])
    print(f"  Reshaped grid: {attention_grid.shape}  (T_chunks, H_merged, W_merged)")

    # 7. Extract frames for visualization
    print(f"\nExtracting frames for visualization...")
    frames = extract_frames_from_video_input(video_inputs)
    print(f"  Got {len(frames)} frames")

    # 8. Render
    print(f"\nRendering visualization to {args.output_dir}/")
    render_visualization(
        attention_grid, frames, args.output_dir, p_safe, p_risk, args.video,
    )

    # 9. Save metadata
    meta = {
        "video": args.video,
        "p_safe": p_safe,
        "p_risk": p_risk,
        "prediction": pred,
        "video_grid_thw": inputs.video_grid_thw[0].tolist(),
        "num_visual_tokens": int(vend - vstart),
        "total_tokens": int(input_ids.size(0)),
        "attention_layers_used": args.last_n_layers,
    }
    with open(Path(args.output_dir) / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Done. See {args.output_dir}/overlay.png")


if __name__ == "__main__":
    main()
