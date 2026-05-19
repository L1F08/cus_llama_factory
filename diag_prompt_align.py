"""
验证：单样本推理构造的 prompt 是否与训练时 LlamaFactory 真实喂的 prompt 一致。

训练侧：LlamaFactory 自己的 `qwen3_5_nothink` template → encode_oneturn →
        prompt_ids（= 训练时 label 被 IGNORE 的那段，即真实 prompt）
推理侧：HF processor.apply_chat_template(enable_thinking=False) + strip "<think>\\n\\n</think>\\n\\n"

只比**文本 scaffold**（system / 角色标记 / assistant 起始前缀 / 是否多塞 <think> 块）。
视频 token 由同一个 processor 在两边等同展开，不影响"最后一个 prompt token 相对答案"的位置，
故这里去掉视频部分做纯文本对比，专门盯 template 路径差异。

只读，CPU，无需 NPU。

用法：
  cd /home/ma-user/work/lyf/
  # 按需改 MODEL_PATH / DATA_PATH / LF_SRC / TEMPLATE
  python diag_prompt_align.py
"""

import sys, json
from types import SimpleNamespace

MODEL_PATH = "/home/ma-user/work/lyf/outmodel/crash_1cam_2cls_train_3s_39k_0508-800_nothink-merge"
DATA_PATH = "/home/ma-user/work/lyf/data/0506_crash_1cam_2cls_test_39k_3s/test_0506_crash_1cam_2cls_test_39k_3s_front_with_ego_info_5256_3s_clipped_cleaned_dedup_4k.json"
# 训练用的 LlamaFactory 源码路径（与 cls_head 训练同一份；和 infer 脚本里 sys.path 一致）
LF_SRC = "/home/ma-user/work/lyf/LlamaFactory-qwen35/src"
TEMPLATE = "qwen3_5_nothink"   # 训练 yaml: template: qwen3_5_nothink

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

sys.path.insert(0, LF_SRC)
try:
    from llamafactory.data.template import get_template_and_fix_tokenizer
except Exception as e:
    print(f"无法从 {LF_SRC} 导入 LlamaFactory template: {e!r}")
    print("→ 把 LF_SRC 改成 cls 训练实际用的 LlamaFactory 源码目录再跑。")
    sys.exit(1)

# get_template_and_fix_tokenizer 只读 data_args 这几个属性
data_args = SimpleNamespace(
    template=TEMPLATE,
    train_on_prompt=False,
    tool_format=None,
    default_system=None,
    enable_thinking=False,     # 训练 yaml: enable_thinking: false
    preserve_thinking=False,
)
template = get_template_and_fix_tokenizer(tok, data_args)
print(f"LlamaFactory template = {TEMPLATE} (class={type(template).__name__}, "
      f"enable_thinking={getattr(template,'enable_thinking',None)})")


def extract_text_and_system(sample):
    """从测试 JSON 一条样本里抽出 (user_text, system_text)。去掉视频部分。"""
    msgs = sample if isinstance(sample, list) else [sample]
    system_text = None
    user_text = None
    for m in msgs:
        role = m.get("role")
        content = m.get("content")
        if isinstance(content, list):
            texts = [c.get("text", "") for c in content if c.get("type") == "text"]
            joined = "\n".join(t for t in texts if t)
        else:
            joined = content or ""
        if role == "system":
            system_text = joined
        elif role == "user":
            user_text = joined
    return user_text or "(empty user text)", system_text


sample = json.load(open(DATA_PATH))[0]
user_text, system_text = extract_text_and_system(sample)
DUMMY_ANSWER = "高风险"  # 答案内容不影响 prompt_ids（prompt = 除最后一条外全部）

# ---------- 训练侧：LlamaFactory encode_oneturn ----------
lf_messages = [
    {"role": "user", "content": user_text},
    {"role": "assistant", "content": DUMMY_ANSWER},
]
prompt_ids, _resp_ids = template.encode_oneturn(tok, lf_messages, system=system_text)
train_prompt_str = tok.decode(prompt_ids, skip_special_tokens=False)

# ---------- 推理侧：HF apply_chat_template + strip ----------
hf_messages = []
if system_text:
    hf_messages.append({"role": "system", "content": [{"type": "text", "text": system_text}]})
hf_messages.append({"role": "user", "content": [{"type": "text", "text": user_text}]})
infer_prompt_str = tok.apply_chat_template(
    hf_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
)
infer_prompt_str_stripped = infer_prompt_str.replace("<think>\n\n</think>\n\n", "")
infer_ids = tok(infer_prompt_str_stripped, add_special_tokens=False)["input_ids"]

# ---------- 对比 ----------
print("\n========== 训练侧 prompt（LlamaFactory qwen3_5_nothink, 末尾 200 字符）==========")
print(repr(train_prompt_str[-200:]))
print(f"  len(prompt_ids) = {len(prompt_ids)}")

print("\n========== 推理侧 prompt（apply_chat_template+strip 前, 末尾 200 字符）==========")
print(repr(infer_prompt_str[-200:]))
print("\n========== 推理侧 prompt（strip 后, 末尾 200 字符）==========")
print(repr(infer_prompt_str_stripped[-200:]))
print(f"  len(infer_ids) = {len(infer_ids)}")

# token-id 前缀匹配长度
n = min(len(prompt_ids), len(infer_ids))
match = 0
for i in range(n):
    if prompt_ids[i] == infer_ids[i]:
        match += 1
    else:
        break

print("\n========== 判定 ==========")
str_same = (train_prompt_str == infer_prompt_str_stripped)
ids_same = (list(prompt_ids) == list(infer_ids))
print(f"字符串完全一致 : {str_same}")
print(f"token ids 完全一致 : {ids_same}  (前缀匹配 {match}/{n})")

if ids_same:
    print(">>> ★ 完全对齐：单样本推理 prompt 与训练逐 token 一致 → 单样本版确定完全正确，可作金标准。")
else:
    print(">>> ✗ 不一致。下面给出首个分歧点，便于对齐推理侧 prompt 构造：")
    if match < n:
        a = prompt_ids[match]
        b = infer_ids[match]
        ctx_tr = tok.decode(prompt_ids[max(0, match - 8):match + 4], skip_special_tokens=False)
        ctx_if = tok.decode(infer_ids[max(0, match - 8):match + 4], skip_special_tokens=False)
        print(f"  首个不同 token @idx {match}:")
        print(f"    训练: id={a} ({tok.decode([a])!r})  上下文: {ctx_tr!r}")
        print(f"    推理: id={b} ({tok.decode([b])!r})  上下文: {ctx_if!r}")
    print(f"  长度: 训练 {len(prompt_ids)} vs 推理 {len(infer_ids)}")
    print("  → 把以上输出贴回，针对差异调整推理侧 prompt（strip 规则 / 模板参数 / system）。")
