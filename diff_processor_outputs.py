"""
对比 Qwen3.5-VL processor 在两种调用模式下的输出结构：
  (A) 真实 batched 调用：processor(text=[t0..tN], images=..., videos=..., padding=True)
  (B) N 个 per-sample 调用 (padding=False)

把两边的字段名、shape、dtype 全部 dump 出来，方便我对照后写出正确的 collator。
不依赖 NPU，纯 CPU。

用法：
  cd /home/ma-user/work/lyf/
  # 改下面的 MODEL_PATH 和 DATA_PATH
  python diff_processor_outputs.py
"""

import os
import json
import torch
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

# ====== 改成你的实际路径 ======
MODEL_PATH = "/home/ma-user/work/lyf/outmodel/crash_1cam_2cls_train_3s_39k_0508-800_nothink-merge"
DATA_PATH = "/home/ma-user/work/lyf/data/0506_crash_1cam_2cls_test_39k_3s/test_0506_crash_1cam_2cls_test_39k_3s_front_with_ego_info_5256_3s_clipped_cleaned_dedup_4k.json"
# =============================

N = 4  # 小批量足够暴露差异

processor = AutoProcessor.from_pretrained(MODEL_PATH)
processor.tokenizer.padding_side = "left"

with open(DATA_PATH) as f:
    samples = json.load(f)[:N]


def prep(sample):
    """复刻 preprocess_sample 的前半段，返回 processor 调用所需的所有原料"""
    messages = sample if isinstance(sample, list) else [sample]
    for msg in messages:
        if msg.get("role") == "user":
            for c in msg.get("content", []):
                if c.get("type") == "video":
                    c["fps"] = 8.0
                    c["max_pixels"] = 589824

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    text = text.replace("<think>\n\n</think>\n\n", "")

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True, image_patch_size=16, return_video_metadata=True,
    )

    video_metadatas = None
    if video_inputs and isinstance(video_inputs, (tuple, list)):
        video_inputs, video_metadatas = zip(*video_inputs)
        video_inputs = list(video_inputs)
        video_metadatas = list(video_metadatas)

    return text, image_inputs, video_inputs, video_metadatas, video_kwargs or {}


def describe(name, obj):
    """把 BatchFeature / dict 里所有字段都打印出来"""
    print(f"\n--- {name} ---")
    for k, v in obj.items():
        if torch.is_tensor(v):
            print(f"  {k}: tensor shape={tuple(v.shape)}, dtype={v.dtype}")
        elif isinstance(v, list):
            inner = type(v[0]).__name__ if v else "?"
            extra = ""
            if v and torch.is_tensor(v[0]):
                extra = f", first.shape={tuple(v[0].shape)}, first.dtype={v[0].dtype}"
            elif v and isinstance(v[0], list):
                extra = f", first.len={len(v[0])}"
            print(f"  {k}: list len={len(v)}, inner={inner}{extra}")
        elif isinstance(v, (int, float, bool, str)):
            print(f"  {k}: {type(v).__name__}={v!r}")
        else:
            print(f"  {k}: {type(v).__name__}={v!r}")


# ===== (A) 真实批量调用 =====
texts, all_images, all_videos, all_metadatas = [], [], [], []
per_sample_kwargs = []
for s in samples:
    text, ii, vi, vm, vk = prep(s)
    texts.append(text)
    if ii: all_images.extend(ii)
    if vi: all_videos.extend(vi)
    if vm: all_metadatas.extend(vm)
    per_sample_kwargs.append(vk)

# 合并 video_kwargs
all_kwargs = {}
all_kw_keys = set()
for vk in per_sample_kwargs:
    all_kw_keys.update(vk.keys())
for k in all_kw_keys:
    vals = [vk[k] for vk in per_sample_kwargs if k in vk]
    if isinstance(vals[0], (list, tuple)):
        flat = []
        for v in vals:
            flat.extend(v)
        all_kwargs[k] = flat
    else:
        all_kwargs[k] = vals[0]

inputs_batched = processor(
    text=texts,
    images=all_images or None,
    videos=all_videos or None,
    video_metadata=all_metadatas or None,
    padding=True,
    return_tensors="pt",
    **all_kwargs,
)

print(f"========== (A) BATCHED call (text=[{N} texts], padding=True) ==========")
describe(f"inputs_batched (batch={N})", inputs_batched)


# ===== (B) per-sample 调用 =====
per_sample_inputs = []
for i, s in enumerate(samples):
    text, ii, vi, vm, vk = prep(s)
    out = processor(
        text=[text],
        images=ii,
        videos=vi,
        video_metadata=vm,
        padding=False,
        return_tensors="pt",
        **vk,
    )
    per_sample_inputs.append(out)

print(f"\n========== (B) PER-SAMPLE calls (N={N}, padding=False) ==========")
for i, ps in enumerate(per_sample_inputs):
    describe(f"sample[{i}]", ps)


# ===== 字段差异对照 =====
batched_keys = set(inputs_batched.keys())
per_sample_keys = set()
for ps in per_sample_inputs:
    per_sample_keys.update(ps.keys())

print("\n========== KEY SET DIFF ==========")
print(f"only in BATCHED:    {sorted(batched_keys - per_sample_keys)}")
print(f"only in PER-SAMPLE: {sorted(per_sample_keys - batched_keys)}")
print(f"in both:            {sorted(batched_keys & per_sample_keys)}")

# 对每个共有字段，比一下 shape / dtype
print("\n========== SHARED-KEY SHAPE/DTYPE ==========")
for k in sorted(batched_keys & per_sample_keys):
    bv = inputs_batched[k]
    pvs = [ps[k] for ps in per_sample_inputs if k in ps]
    if torch.is_tensor(bv):
        print(f"  {k}:")
        print(f"    BATCHED:    shape={tuple(bv.shape)}, dtype={bv.dtype}")
        for i, pv in enumerate(pvs):
            print(f"    SAMPLE[{i}]: shape={tuple(pv.shape)}, dtype={pv.dtype}")
    elif isinstance(bv, list):
        print(f"  {k}:")
        print(f"    BATCHED:    list len={len(bv)}")
        for i, pv in enumerate(pvs):
            print(f"    SAMPLE[{i}]: list len={len(pv) if hasattr(pv, '__len__') else '?'}")
    else:
        print(f"  {k}: BATCHED={bv!r} | PER-SAMPLE[0]={pvs[0]!r}")
