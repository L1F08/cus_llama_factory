"""
验证假设：批量版改调「内层 backbone」绕过了顶层的 mrope position_ids 计算，
导致结果对 batch/padding 敏感（非确定根因）。

对同一个样本，比较 4 条路径的 cls_logits：
  P1 顶层 model(**inp, output_hidden_states=True) → hidden_states[-1][:, -1]   (训练/单样本同款，基准)
  P2 内层 model.model(**inp).last_hidden_state[:, -1]                          (当前批量版 batch=1)
  P3 内层，但与一个更长的 dummy 右 padding 拼成 batch=2，取样本 0 末真实 token (当前批量版真实场景)
  P4 顶层，同样拼 batch=2，取样本 0 末真实 token                              (顶层是否 batch 不变)

判定：
  P1==P2 且 P1==P4 但 P1!=P3  → 证实：内层路径 + 批 padding 是根因
  P1!=P2                       → 内层路径本身就错（与顶层不等价）
  全部相等                     → 路径不是根因，另查

只读诊断，单卡。
"""

import os, sys, json
import torch
import torch_npu  # noqa

MODEL_PATH = "/home/ma-user/work/lyf/outmodel/crash_1cam_2cls_train_3s_39k_0508-800_nothink-merge"
DATA_PATH = "/home/ma-user/work/lyf/data/0506_crash_1cam_2cls_test_39k_3s/test_0506_crash_1cam_2cls_test_39k_3s_front_with_ego_info_5256_3s_clipped_cleaned_dedup_4k.json"
ATTN_IMPL = os.environ.get("ATTN_IMPL", "sdpa")

from pathlib import Path
from transformers import AutoProcessor
try:
    from transformers import Qwen3_5VLForConditionalGeneration as TargetVLModel
except ImportError:
    from transformers import AutoModelForImageTextToText as TargetVLModel
from qwen_vl_utils import process_vision_info
sys.path.insert(0, "/home/ma-user/work/lyf/LlamaFactory-qwen35/src")
from llamafactory.model.cls_head import BinaryClassificationHead  # noqa

device = "npu:0"
torch.npu.set_device(device)
model = TargetVLModel.from_pretrained(
    MODEL_PATH, torch_dtype="auto", device_map=None, attn_implementation=ATTN_IMPL,
).eval().to(device)
processor = AutoProcessor.from_pretrained(MODEL_PATH)
processor.tokenizer.padding_side = "right"
pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id

meta = json.load(open(Path(MODEL_PATH) / "cls_head_meta.json", encoding="utf-8"))
cls_head = BinaryClassificationHead(meta["hidden_size"], dropout=meta.get("dropout", 0.0))
cls_head.load_state_dict(torch.load(Path(MODEL_PATH) / "cls_head.bin", map_location="cpu"))
cls_head = cls_head.to(device).to(torch.float32).eval()
hd = next(cls_head.parameters()).dtype

samples = json.load(open(DATA_PATH))


def prep(sample):
    msgs = json.loads(json.dumps(sample if isinstance(sample, list) else [sample]))
    for m in msgs:
        if m.get("role") == "user":
            for c in m.get("content", []):
                if c.get("type") == "video":
                    c["fps"] = 8.0
                    c["max_pixels"] = 602112
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    text = text.replace("<think>\n\n</think>\n\n", "")
    ii, vi, vk = process_vision_info(msgs, return_video_kwargs=True, image_patch_size=16, return_video_metadata=True)
    vm = None
    if vi and isinstance(vi, (tuple, list)):
        vi, vm = zip(*vi); vi = list(vi); vm = list(vm)
    return processor(text=[text], images=ii, videos=vi, video_metadata=vm,
                     padding=False, return_tensors="pt", **(vk or {}))


# 先扫前 N 个样本拿到各自长度，挑：s0 = 较短的被测样本，dummy = 更长的样本
# （这样 s0 在 batch=2 里会被右 padding，才能测出 padding 是否影响 s0 的结果）
SCAN_N = 40
prepped = []
for s in samples[:SCAN_N]:
    c = prep(s)
    prepped.append((c["input_ids"].shape[1], c))
prepped.sort(key=lambda x: x[0])
len_short, inp0 = prepped[0]          # 最短的当被测样本 s0
len_long, inp_long = prepped[-1]      # 最长的当 dummy
if len_long <= len_short:
    inp_long = None
print(f"扫描 {SCAN_N} 个样本：长度范围 [{prepped[0][0]} .. {prepped[-1][0]}]")
print(f"s0(被测,最短) len={len_short}, dummy(最长) len={len_long} "
      f"→ s0 在 batch=2 会被右 pad {max(0, len_long-len_short)} 个 token")


def head_logits(h):
    return cls_head(h.to(hd)).float().cpu().tolist()


def P1_toplevel_b1(cpu):
    inp = cpu.to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True, return_dict=True, use_cache=False)
    h = out.hidden_states[-1][:, -1, :]
    return head_logits(h)


def P2_inner_b1(cpu):
    inp = cpu.to(device)
    with torch.no_grad():
        out = model.model(**inp, return_dict=True, use_cache=False)
    h = out.last_hidden_state[:, -1, :]
    return head_logits(h)


def _right_pad_pair(a, b):
    """把 a,b 两个 batch=1 BatchFeature 右 padding 拼成 batch=2（仿 collator）。
    注意：BatchFeature.to(device) 是原地修改，前面 P1/P2 可能已把 a 的 tensor
    挪到 NPU。这里统一 .cpu() 后在 CPU 上拼，最后由调用方 .to(device)。"""
    from transformers.feature_extraction_utils import BatchFeature
    maxlen = max(a["input_ids"].shape[1], b["input_ids"].shape[1])
    out = {}
    keys = set(a.keys()) | set(b.keys())
    for k in keys:
        va, vb = a[k], b[k]
        if torch.is_tensor(va):
            va = va.cpu()
        if torch.is_tensor(vb):
            vb = vb.cpu()
        if torch.is_tensor(va) and va.dim() >= 2 and va.shape[0] == 1 and va.shape[1] != vb.shape[1]:
            pv = pad_id if k == "input_ids" else 0
            def rp(v):
                if v.shape[1] < maxlen:
                    ps = list(v.shape); ps[1] = maxlen - v.shape[1]
                    v = torch.cat([v, torch.full(ps, pv, dtype=v.dtype)], dim=1)
                return v
            out[k] = torch.cat([rp(va), rp(vb)], dim=0)
        elif torch.is_tensor(va):
            out[k] = torch.cat([va, vb], dim=0)
        else:
            out[k] = [va, vb]
    return BatchFeature(data=out)


def P3_inner_b2(a, b):
    inp = _right_pad_pair(a, b).to(device)
    with torch.no_grad():
        out = model.model(**inp, return_dict=True, use_cache=False)
    lh = out.last_hidden_state
    last0 = inp["attention_mask"][0].sum() - 1
    h = lh[0:1, last0, :]
    return head_logits(h)


def P4_toplevel_b2(a, b):
    inp = _right_pad_pair(a, b).to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True, return_dict=True, use_cache=False)
    lh = out.hidden_states[-1]
    last0 = inp["attention_mask"][0].sum() - 1
    h = lh[0:1, last0, :]
    return head_logits(h)


def _compute_position_ids(inp):
    """复刻 generate 的 mrope position_ids 准备。尝试多个 API，返回 (position_ids, how) 或 (None, err)。"""
    # 优先：顶层模型的 _prepare_position_ids_for_generation (generate 内部就是它)
    fn = getattr(model, "_prepare_position_ids_for_generation", None)
    if fn is not None:
        try:
            mk = {k: v for k, v in inp.items() if k != "input_ids"}
            pos = fn(inp["input_ids"], mk)
            return pos, "_prepare_position_ids_for_generation"
        except Exception as e:
            last = f"_prepare_position_ids_for_generation failed: {e!r}"
    else:
        last = "_prepare_position_ids_for_generation absent"
    # 退路：内层 get_rope_index
    gri = getattr(getattr(model, "model", None), "get_rope_index", None) or getattr(model, "get_rope_index", None)
    if gri is not None:
        try:
            res = gri(
                inp["input_ids"],
                image_grid_thw=inp.get("image_grid_thw"),
                video_grid_thw=inp.get("video_grid_thw"),
                attention_mask=inp.get("attention_mask"),
            )
            pos = res[0] if isinstance(res, (tuple, list)) else res
            return pos, "get_rope_index"
        except Exception as e:
            last = f"{last}; get_rope_index failed: {e!r}"
    return None, last


def P5_inner_b2_explicit_pos(a, b):
    """batch=2 右pad，但显式算好 mrope position_ids 再传给内层 forward，取样本 0"""
    inp = _right_pad_pair(a, b).to(device)
    pos, how = _compute_position_ids(inp)
    if pos is None:
        return None, how
    with torch.no_grad():
        out = model.model(**inp, position_ids=pos, return_dict=True, use_cache=False)
    lh = out.last_hidden_state
    last0 = inp["attention_mask"][0].sum() - 1
    h = lh[0:1, last0, :]
    return head_logits(h), how


p1 = P1_toplevel_b1(inp0)
p2 = P2_inner_b1(inp0)
print(f"\nP1 顶层 batch=1            : {p1}")
print(f"P2 内层 batch=1            : {p2}")
if inp_long is not None:
    p3 = P3_inner_b2(inp0, inp_long)
    p4 = P4_toplevel_b2(inp0, inp_long)
    print(f"P3 内层 batch=2 (右pad)    : {p3}")
    print(f"P4 顶层 batch=2 (右pad)    : {p4}")
    p5, p5how = P5_inner_b2_explicit_pos(inp0, inp_long)
    print(f"P5 内层 b=2 右pad +显式pos : {p5}   [{p5how}]")
else:
    p3 = p4 = p5 = None
    print("（没找到更长的 dummy，跳过 P3/P4/P5）")


def close(x, y, tol=1e-3):
    return x is not None and y is not None and abs(x[0][0]-y[0][0]) < tol and abs(x[0][1]-y[0][1]) < tol

print("\n===== 判定 =====")
print(f"P1==P2 (内层 vs 顶层, batch=1)        : {close(p1,p2)}")
if p3 is not None:
    print(f"P1==P3 (右pad 是否改变结果?)         : {close(p1,p3)}  (False=右pad 确实改了)")
    print(f"P1==P5 (显式 position_ids 能否修复?) : {close(p1,p5)}  (True=修复方向确认)")
    if p5 is None:
        print(">>> position_ids 计算 API 没跑通，需要换 API（看上面 [..] 里的报错）。")
    elif close(p1, p5):
        print(">>> ★ 确认根因+解法：forward 没传 mrope position_ids → 对 padding 敏感。"
              "显式算好 position_ids 传入即可修复（与训练/generate 对齐）。")
    elif not close(p1, p3):
        print(">>> 显式 position_ids 仍没修复，padding 影响来自别处（attention_mask 处理 / "
              "vision 融合），需进一步查。")
    else:
        print(">>> 右pad 本身没改变结果，非确定来自别处。")
