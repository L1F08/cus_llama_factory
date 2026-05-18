"""
定位 cls_head 推理 run-to-run 不一致的根源。

分两步独立验证：
  A. 同一样本跑两次 preprocess（含 process_vision_info 解码 + processor），
     比较 input_ids / pixel_values_videos 是否逐元素相同
     → 不同则视频解码/预处理非确定 (根因 A)
  B. 用第一次的 cpu_inputs（固定输入），跑两次 backbone forward + cls_head，
     比较 last_hidden / cls_logits 的最大绝对差
     → 不同则 NPU bf16 kernel 非确定 (根因 B)

只读诊断，不依赖也不修改推理流水线。单卡即可。

用法：
  cd /home/ma-user/work/lyf/
  # 改 MODEL_PATH / DATA_PATH 为你的实际路径
  python diag_cls_head_determinism.py
"""

import os
import sys
import json
import torch
import torch_npu  # noqa: F401

# ====== 改成你的实际路径 ======
MODEL_PATH = "/home/ma-user/work/lyf/outmodel/crash_1cam_2cls_train_3s_39k_0508-800_nothink-merge"
DATA_PATH = "/home/ma-user/work/lyf/data/0506_crash_1cam_2cls_test_39k_3s/test_0506_crash_1cam_2cls_test_39k_3s_front_with_ego_info_5256_3s_clipped_cleaned_dedup_4k.json"
ATTN_IMPL = os.environ.get("ATTN_IMPL", "sdpa")
VIDEO_FPS = float(os.getenv("INFER_VIDEO_FPS", "8.0"))
VIDEO_MAX_PIXELS = int(os.getenv("INFER_VIDEO_MAX_PIXELS", "602112"))
# =============================

from pathlib import Path
from transformers import AutoProcessor

try:
    from transformers import Qwen3_5VLForConditionalGeneration as TargetVLModel
except ImportError:
    from transformers import AutoModelForImageTextToText as TargetVLModel

from qwen_vl_utils import process_vision_info

sys.path.insert(0, "/home/ma-user/work/lyf/LlamaFactory-qwen35/src")
from llamafactory.model.cls_head import BinaryClassificationHead  # noqa: E402

device = "npu:0"
torch.npu.set_device(device)

print(f"loading model (ATTN_IMPL={ATTN_IMPL}) ...")
model = TargetVLModel.from_pretrained(
    MODEL_PATH, torch_dtype="auto", device_map=None, attn_implementation=ATTN_IMPL,
).eval().to(device)
processor = AutoProcessor.from_pretrained(MODEL_PATH)

meta = json.load(open(Path(MODEL_PATH) / "cls_head_meta.json", encoding="utf-8"))
cls_head = BinaryClassificationHead(meta["hidden_size"], dropout=meta.get("dropout", 0.0))
cls_head.load_state_dict(torch.load(Path(MODEL_PATH) / "cls_head.bin", map_location="cpu"))
cls_head = cls_head.to(device).to(torch.float32).eval()
head_dtype = next(cls_head.parameters()).dtype
print(f"model dtype = {next(model.parameters()).dtype}, cls_head dtype = {head_dtype}")

sample = json.load(open(DATA_PATH))[0]
messages = sample if isinstance(sample, list) else [sample]


def preprocess_once():
    msgs = json.loads(json.dumps(messages))  # deep copy（避免 fps/max_pixels 被原地改）
    for m in msgs:
        if m.get("role") == "user":
            for c in m.get("content", []):
                if c.get("type") == "video":
                    c["fps"] = VIDEO_FPS
                    c["max_pixels"] = VIDEO_MAX_PIXELS
    text = processor.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    text = text.replace("<think>\n\n</think>\n\n", "")
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        msgs, return_video_kwargs=True, image_patch_size=16, return_video_metadata=True
    )
    video_metadatas = None
    if video_inputs and isinstance(video_inputs, (tuple, list)):
        video_inputs, video_metadatas = zip(*video_inputs)
        video_inputs = list(video_inputs)
        video_metadatas = list(video_metadatas)
    return processor(
        text=[text], images=image_inputs, videos=video_inputs,
        video_metadata=video_metadatas, padding=False, return_tensors="pt",
        **(video_kwargs or {}),
    )


# ========== 步骤 A：预处理确定性 ==========
print("\n===== A. preprocess determinism (run twice, compare inputs) =====")
inp1 = preprocess_once()
inp2 = preprocess_once()

def cmp_tensor(name, a, b):
    if a.shape != b.shape:
        print(f"  {name}: SHAPE differs {tuple(a.shape)} vs {tuple(b.shape)}  ← 非确定")
        return False
    if a.dtype.is_floating_point:
        d = (a.float() - b.float()).abs().max().item()
        same = torch.equal(a, b)
        print(f"  {name}: shape={tuple(a.shape)} max_abs_diff={d:.3e} identical={same}")
        return same
    else:
        same = torch.equal(a, b)
        print(f"  {name}: shape={tuple(a.shape)} identical={same}")
        return same

a_ok = True
for k in inp1.keys():
    if torch.is_tensor(inp1[k]):
        a_ok &= cmp_tensor(k, inp1[k], inp2[k])
print(f"--> 预处理{'确定 (A 排除)' if a_ok else '不确定 ★ 根因是 A：视频解码/预处理'}")

# ========== 步骤 B：固定输入下 forward 确定性 ==========
print("\n===== B. forward determinism (same fixed inputs, run twice) =====")

def forward_once(cpu_inputs):
    inputs = cpu_inputs.to(device)
    with torch.no_grad():
        backbone = getattr(model, "model", None)
        if backbone is not None:
            out = backbone(**inputs, return_dict=True, use_cache=False)
            last_hidden = out.last_hidden_state
        else:
            out = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
            last_hidden = out.hidden_states[-1]
    h = last_hidden[:, -1, :]                      # batch=1，最后 token
    cls_logits = cls_head(h.to(head_dtype))
    return last_hidden.float().cpu(), cls_logits.float().cpu()

lh1, lg1 = forward_once(inp1)
lh2, lg2 = forward_once(inp1)   # 注意：两次都用 inp1（同一份固定输入）

dh = (lh1 - lh2).abs().max().item()
dl = (lg1 - lg2).abs().max().item()
print(f"  last_hidden max_abs_diff = {dh:.3e}")
print(f"  cls_logits  max_abs_diff = {dl:.3e}")
print(f"  run1 logits = {lg1.tolist()}")
print(f"  run2 logits = {lg2.tolist()}")
b_ok = (dh == 0.0 and dl == 0.0)
print(f"--> forward{'确定 (B 排除)' if b_ok else ' 不确定 ★ 根因是 B：NPU bf16 kernel'}")

print("\n===== 结论 =====")
if not a_ok:
    print("根因 A：视频解码非确定。换 decord 解码 / 固定抽帧索引 可解。")
elif not b_ok:
    print("根因 B：bf16 backbone + NPU kernel 非确定。")
    print("  - 完全确定：ATTN_IMPL=eager 再测；或把 backbone 也跑 fp32（慢/吃显存）")
    print("  - 或接受：只有贴边样本翻转，与训练时同源（训练也是 bf16+非确定）")
else:
    print("A、B 都确定。非确定性来自别处（如 batch 组成 → 仍需排查 collator/padding）。")
