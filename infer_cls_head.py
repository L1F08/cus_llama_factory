"""Generic inference for binary classification head.

Works for any binary task — class names are read from the checkpoint's
cls_head_meta.json (positive_text / negative_text), not hardcoded.

Usage:
    python infer_cls_head.py \
        --base_model /home/ma-user/work/lyf/model/Qwen3_5-9B \
        --ckpt /path/to/checkpoint-XXXX \
        --manifest test_set.jsonl \
        --output preds.jsonl

manifest.jsonl format (one per line):
    {"video_path": "...", "prompt": "...", "label": "<positive_text>" | "<negative_text>"}
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

# Make sure we can import llamafactory.model.cls_head
sys.path.insert(0, "/home/ma-user/work/lyf/LlamaFactory-qwen35/src")
from llamafactory.model.cls_head import BinaryClassificationHead  # noqa: E402

from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer  # noqa: E402


@torch.no_grad()
def predict_one(
    model,
    processor,
    head,
    prompt_text,
    video_path,
    label_map,
    video_max_pixels=589824,
    video_fps=10.0,
    video_maxlen=32,
    device="npu:0",
):
    """Run forward and return per-class probabilities.

    label_map: {"0": <negative_text>, "1": <positive_text>}
    Output keys are class-agnostic: p_class_0, p_class_1, pred (text), pred_class (0/1).
    """
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": video_max_pixels,
                    "fps": video_fps,
                    "max_frames": video_maxlen,
                },
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    last_hidden = outputs.hidden_states[-1]              # [1, T, D]
    h = last_hidden[:, -1, :]                            # last position before generation
    head_dtype = next(head.parameters()).dtype
    cls_logits = head(h.to(head_dtype))                  # [1, 2]
    probs = torch.softmax(cls_logits.float(), dim=-1)    # [1, 2]
    p0 = probs[0, 0].item()
    p1 = probs[0, 1].item()
    pred_class = 1 if p1 >= 0.5 else 0
    return {
        "p_class_0": p0,
        "p_class_1": p1,
        "p_positive": p1,                                # alias for class 1
        "p_negative": p0,                                # alias for class 0
        "pred_class": pred_class,
        "pred": label_map[str(pred_class)],
        "logit_class_0": cls_logits[0, 0].item(),
        "logit_class_1": cls_logits[0, 1].item(),
    }


def load_model_with_head(base_model_path, ckpt_path, device="npu:0"):
    print(f"Loading tokenizer/processor from {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)

    print(f"Loading base model from {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    print(f"Loading LoRA + modules_to_save from {ckpt_path}")
    model = PeftModel.from_pretrained(base_model, ckpt_path)
    model.eval()

    print(f"Loading classification head from {ckpt_path}/cls_head.bin")
    meta_path = Path(ckpt_path) / "cls_head_meta.json"
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    head = BinaryClassificationHead(meta["hidden_size"], dropout=meta.get("dropout", 0.1))
    head.load_state_dict(torch.load(Path(ckpt_path) / "cls_head.bin", map_location="cpu"))
    head = head.to(device).to(torch.bfloat16)
    head.eval()

    model = model.to(device)

    # Backward-compat: older meta might use high_risk_*/safe_* keys
    if "label_map" not in meta:
        meta["label_map"] = {
            "0": meta.get("negative_text", meta.get("safe_text", "class_0")),
            "1": meta.get("positive_text", meta.get("high_risk_text", "class_1")),
        }
    print(
        f"Model + head ready on {device}. "
        f"label_map: 0='{meta['label_map']['0']}', 1='{meta['label_map']['1']}'"
    )
    return model, processor, tokenizer, head, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="/home/ma-user/work/lyf/model/Qwen3_5-9B")
    parser.add_argument("--ckpt", required=True, help="Checkpoint dir (adapter_model.* + cls_head.bin)")
    parser.add_argument("--manifest", required=True, help="JSONL with {video_path, prompt, label?}")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--video_max_pixels", type=int, default=589824)
    parser.add_argument("--video_fps", type=float, default=10.0)
    parser.add_argument("--video_maxlen", type=int, default=32)
    args = parser.parse_args()

    model, processor, _tokenizer, head, meta = load_model_with_head(
        args.base_model, args.ckpt, device=args.device
    )
    label_map = meta["label_map"]

    with open(args.manifest, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(items)} samples from {args.manifest}")

    results = []
    for i, item in enumerate(items):
        try:
            out = predict_one(
                model, processor, head,
                prompt_text=item["prompt"],
                video_path=item["video_path"],
                label_map=label_map,
                video_max_pixels=args.video_max_pixels,
                video_fps=args.video_fps,
                video_maxlen=args.video_maxlen,
                device=args.device,
            )
        except Exception as e:
            out = {
                "error": str(e),
                "p_class_0": -1.0,
                "p_class_1": -1.0,
                "pred": "ERROR",
                "pred_class": -1,
            }
        out["video_path"] = item["video_path"]
        out["label"] = item.get("label")
        results.append(out)
        if (i + 1) % 50 == 0 or (i + 1) == len(items):
            print(f"[{i + 1}/{len(items)}] last p_class_1={out.get('p_class_1', -1):.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(results)} predictions to {args.output}")


if __name__ == "__main__":
    main()
