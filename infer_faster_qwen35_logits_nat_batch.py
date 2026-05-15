"""
Qwen3.5-VL logits_nat 推理的批量版本。
基于 infer_faster_qwen35_logits_nat.py：
  - 只取第一步 logits (max_new_tokens=1)，提取 "安全" / "高风险" 两个 token 的分数
  - strip 模板里的空 <think>\\n\\n</think>\\n\\n 块，与训练 prompt 对齐
改造点：把单样本 + 全局锁的串行推理，换成 CPU 预处理多线程 + NPU 批量 generate 的流水线。
"""

import os
import json
import heapq
import time
import threading
import traceback
import queue as _queue
from tqdm import tqdm
import torch
import torch_npu
import torch.distributed as dist
from concurrent.futures import ThreadPoolExecutor, as_completed

# [Qwen3.5 通用引入]
try:
    from transformers import Qwen3_5VLForConditionalGeneration as TargetVLModel, AutoProcessor, HfArgumentParser
except ImportError:
    # 强烈建议使用 AutoModelForImageTextToText 作为兜底，可自动路由 Dense 或 MoE
    from transformers import AutoModelForImageTextToText as TargetVLModel, AutoProcessor, HfArgumentParser

from infer_params import DataArguments, ModelArguments, TrainingArguments
from qwen_vl_utils import process_vision_info

# ====================== 配置 ======================
# CPU 预处理并行线程数 (视频解码 + chat_template + process_vision_info)
PREPROCESS_WORKERS = int(os.environ.get("PREPROCESS_WORKERS", 6))

# NPU 单次批大小。logits_nat 模式 max_new_tokens=1，KV cache 只有 prefill，
# 显存开销远低于 thinking 模式，BATCH_SIZE 可以开得更大。
# 910B 65GB 单卡推荐 8 起步，富裕时上调到 12/16。
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 8))

torch.npu.config.allow_internal_format = False
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"


# ====================== CPU 预处理 ======================
def preprocess_sample(processor, messages, video_path):
    """单样本 CPU 预处理，与原 logits_nat 单样本版本严格对齐。"""
    try:
        for msg in messages:
            if msg.get("role") == "user":
                for content_item in msg.get("content", []):
                    if content_item.get("type") == "video":
                        content_item["fps"] = 8.0
                        # 与训练对齐：max_pixels = 589824
                        content_item["max_pixels"] = 589824

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        # 关键：strip 掉模板硬塞的空 <think> 块，使推理 prompt 与训练完全一致
        # (Qwen3.5-VL chat_template 即使 enable_thinking=False 也会插入 "<think>\n\n</think>\n\n")
        text = text.replace("<think>\n\n</think>\n\n", "")

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

        # ★ 关键：把 processor() 调用挪到 worker 线程里，每样本一次。
        # 上次单进程 bench 显示，跨 8 ranks 并发时 batched processor 比 single-sample
        # 慢得多 (23s vs 14.6s)；并行做 single-sample 再 collate 才是正确架构。
        cpu_inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            video_metadata=video_metadatas,
            padding=False,                    # 单样本无需 padding
            return_tensors="pt",
            **(video_kwargs or {}),
        )

        return {
            "video_path": video_path,
            "cpu_inputs": cpu_inputs,
            "error": None,
        }
    except Exception as e:
        return {"video_path": video_path, "error": str(e)}


# ====================== Custom collator (替代主线程的 batched processor 调用) ======================
def _collate_qwen_vl(per_sample_inputs, pad_token_id):
    """把 N 份 batch=1 的 BatchFeature 拼成 batch=N。

    根据 diff_processor_outputs.py 实测，Qwen3.5-VL processor 输出 5 个字段：
      - input_ids / attention_mask / mm_token_type_ids: shape [1, seq_len_i] —— 左 pad 到 max_len
      - pixel_values_videos: shape [num_patches_i, 1536] —— 沿 dim 0 cat
      - video_grid_thw: shape [num_videos_i, 3] —— 沿 dim 0 cat

    采用通用规则: 所有 shape=[1, X] 且 X 跨样本不一致的 tensor 都 left-pad；
    其它 tensor 沿 dim 0 cat；list/标量按列表 extend。这样将来新字段也不会再翻车。
    """
    from transformers.feature_extraction_utils import BatchFeature

    if not per_sample_inputs:
        return None

    all_keys = set()
    for s in per_sample_inputs:
        all_keys.update(s.keys())

    out = {}
    for k in all_keys:
        vals = [s[k] for s in per_sample_inputs if k in s]
        if not vals:
            continue
        v0 = vals[0]

        if torch.is_tensor(v0) and v0.dim() >= 2 and v0.shape[0] == 1:
            # 形如 [1, X, ...] 的 per-sample tensor。
            # **关键**：是否需要 pad 只看「本字段 dim-1 各样本是否一致」——
            # 不能拿 dim-1 跟 input_ids 的 max_len 比，否则会把 video_grid_thw 这种
            # [1, 3] (3 是 T/H/W 三元组，不是 seq_len) 错误地 pad 到 max_len。
            field_lens = [v.shape[1] for v in vals]
            if len(set(field_lens)) > 1:
                field_max = max(field_lens)
                pad_val = pad_token_id if k == "input_ids" else 0
                padded = []
                for v in vals:
                    cur = v.shape[1]
                    if cur < field_max:
                        pad_shape = list(v.shape)
                        pad_shape[1] = field_max - cur
                        pad_t = torch.full(pad_shape, pad_val, dtype=v.dtype, device=v.device)
                        v = torch.cat([pad_t, v], dim=1)  # left-pad
                    padded.append(v)
                out[k] = torch.cat(padded, dim=0)
            else:
                # 各样本 dim-1 一致 → 直接 cat 成 [N, X, ...]
                out[k] = torch.cat(vals, dim=0)
        elif torch.is_tensor(v0):
            # 例如 pixel_values_videos [num_patches, hidden]
            try:
                out[k] = torch.cat(vals, dim=0)
            except Exception:
                out[k] = vals  # 兜底
        elif isinstance(v0, list):
            merged = []
            for v in vals:
                merged.extend(v)
            out[k] = merged
        else:
            out[k] = vals  # 标量等

    return BatchFeature(data=out)


# ====================== NPU 批量推理 + Logits 提取 ======================
class BatchedLogitsInferencer:
    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device

        # ★ left padding：autoregressive 批量生成的硬性要求。
        # 我们只关心「prompt 之后第一个 token」的 logits，left pad 让所有样本的
        # 第一个新 token 落在同一时间步 (outputs.scores[0]) 上。
        self.processor.tokenizer.padding_side = "left"

        # 目标 token id (与原单样本版本一致：取 BPE 拆出来的第一个 token)
        self.token_id_safe = self.processor.tokenizer.encode("安全", add_special_tokens=False)[0]
        self.token_id_risk = self.processor.tokenizer.encode("高风险", add_special_tokens=False)[0]

        pad_id = self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id

        # 与原版完全一致：只取第一步 logits，max_new_tokens=1 已足够，不会改 logit 值
        self.gen_kwargs = {
            "max_new_tokens": 1,
            "do_sample": False,
            "temperature": 0.1,
            "top_p": 0.95,
            "repetition_penalty": 1.00,
            "return_dict_in_generate": True,
            "output_scores": True,
            "pad_token_id": pad_id,
        }

    def build_inputs_cpu(self, batch, _stats=None):
        """主线程 CPU 工作：把 N 份 batch=1 的 per-sample BatchFeature collate 成 batch=N。
        重活 (processor() 调用) 已经在 PREPROCESS_WORKERS 里每样本并行做完，
        主线程只做轻量 left-pad + cat，应该是毫秒级。"""
        if _stats is not None:
            _t0 = time.perf_counter()

        per_sample = [item["cpu_inputs"] for item in batch]
        pad_id = self.gen_kwargs["pad_token_id"]
        inputs = _collate_qwen_vl(per_sample, pad_token_id=pad_id)

        if _stats is not None:
            _stats["collate"] = _stats.get("collate", 0.0) + (time.perf_counter() - _t0)
        return inputs

    def run_inference(self, batch, cpu_inputs):
        """NPU 工作线程调用：把 inputs 搬到 NPU → generate → 提 logits → 解码 → 拼结果。"""
        inputs = cpu_inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **self.gen_kwargs)

        # outputs.scores 是 tuple，长度 = max_new_tokens = 1
        # outputs.scores[0] shape = [batch_size, vocab_size]：第一个新 token 的预测分布
        first_step_logits = outputs.scores[0]  # [B, V]

        # 在 NPU 上一次性提两个 token 的 logits 再搬到 CPU，避免每个样本一次 .item() 的同步开销
        target_ids = torch.tensor(
            [self.token_id_safe, self.token_id_risk],
            device=first_step_logits.device,
            dtype=torch.long,
        )
        target_logits_list = first_step_logits[:, target_ids].float().cpu().tolist()  # [B, 2]

        # 解码生成的那 1 个 token (保留 answers 字段，与原版输出格式兼容)
        prompt_len = inputs.input_ids.shape[1]
        generated_ids_trimmed = outputs.sequences[:, prompt_len:]
        out_texts = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        # 注意：去掉了 torch.npu.empty_cache() —— 每 batch 调一次会插强同步点拖慢节奏，
        # 显存富裕的话完全不需要。

        return [
            {
                "id": item["video_path"],
                "answers": [out_texts[i]],
                "logits": {
                    "安全": target_logits_list[i][0],
                    "高风险": target_logits_list[i][1],
                },
            }
            for i, item in enumerate(batch)
        ]

    def infer_batch(self, batch):
        """同步推理，仅用于主流水线之外的兜底分支。"""
        if not batch:
            return []
        try:
            cpu_inputs = self.build_inputs_cpu(batch)
            return self.run_inference(batch, cpu_inputs)
        except Exception as e:
            err = f"Batch Error: {str(e)}"
            print(f"[Batch fail @ {self.device}] {err}")
            return [
                {"id": item["video_path"], "answers": [err]}
                for item in batch
            ]


# ====================== 主工作流 ======================
def main_worker(rank, world_size, model_args, data_args, training_args):
    device = f'npu:{rank}'
    torch.npu.set_device(device)

    print(f"Rank {rank}: 加载 Qwen3.5 模型 (BATCH_SIZE={BATCH_SIZE}, PREPROC_WORKERS={PREPROCESS_WORKERS})...")
    model = TargetVLModel.from_pretrained(
        model_args.model_id,
        torch_dtype="auto",
        device_map=None,
        attn_implementation="flash_attention_2",
    ).eval().to(device)

    processor = AutoProcessor.from_pretrained(model_args.model_id)
    inferencer = BatchedLogitsInferencer(model, processor, device)

    # 准备输出文件
    output_file = f"{data_args.output_json.rsplit('.', 1)[0]}_rank{rank}.json"

    # 读取本 Rank 已完成结果 (针对当前 Rank 的中间文件)
    completed_ids = set()
    results = []
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
                completed_ids = {r['id'] for r in results}
        except: pass

    # 数据分配 (仅 Rank 0 读取并分发)
    if rank == 0:
        # Rank 0 读取全局最终文件，确保分配时不会重复分配以前跑完的全局数据
        global_completed_ids = set()
        if os.path.exists(data_args.output_json):
            try:
                with open(data_args.output_json, 'r', encoding='utf-8') as f:
                    global_existing = json.load(f)
                    global_completed_ids = {r['id'] for r in global_existing}
            except: pass

        with open(data_args.data_path, 'r', encoding='utf-8') as f:
            test_samples = json.load(f)

        valid_samples = []
        for sample in test_samples:
            try:
                first_msg = sample[0] if isinstance(sample, list) else sample
                content = first_msg.get("content", [])
                video_path = next((item["video"] for item in content if item["type"] == "video"), None)
                if video_path and video_path not in global_completed_ids and video_path not in completed_ids:
                    valid_samples.append((sample, video_path))
            except: continue

        sample_sizes = []
        for sample, video_path in valid_samples:
            size = os.path.getsize(video_path) if os.path.exists(video_path) else 0
            sample_sizes.append((size, sample, video_path))
        sample_sizes.sort(key=lambda x: x[0], reverse=True)

        worker_loads = [(0, w) for w in range(world_size)]
        heapq.heapify(worker_loads)
        # 临时保留 size，方便分配完成后在 rank 内按大小排序
        task_lists_sized = [[] for _ in range(world_size)]

        for size, sample, video_path in sample_sizes:
            load, w = heapq.heappop(worker_loads)
            task_lists_sized[w].append((size, sample, video_path))
            heapq.heappush(worker_loads, (load + size, w))

        # ★ 每个 rank 内部按文件大小降序排序，让相邻样本视频复杂度相近 ——
        # 同一个 batch 内的 token 数差异变小，left-padding 浪费的算力随之减少。
        task_lists = []
        for w in range(world_size):
            task_lists_sized[w].sort(key=lambda x: x[0], reverse=True)
            task_lists.append([(s, v) for _, s, v in task_lists_sized[w]])

        dist_obj = [task_lists]
    else:
        dist_obj = [None]

    # 广播任务
    dist.broadcast_object_list(dist_obj, src=0)
    my_tasks = dist_obj[0][rank]

    # 过滤掉本 Rank 已经跑过的 (Double check)
    my_queue = [t for t in my_tasks if t[1] not in completed_ids]

    print(f"Rank {rank}: 分配到 {len(my_queue)} 个任务，预处理并行数 {PREPROCESS_WORKERS}，推理 batch={BATCH_SIZE}")

    # ========== 三段流水线 ==========
    # Stage 1 (CPU 线程池, PREPROCESS_WORKERS 个): 视频解码 + chat_template + vision tokenize
    # Stage 2 (主线程):                          凑齐 BATCH_SIZE → build_inputs_cpu (processor 打包)
    # Stage 3 (NPU 单线程):                       .to(device) → model.generate → 提 logits + decode
    # Stage 2 和 Stage 3 通过 npu_inbox/npu_outbox 队列**异步**衔接 ——
    # 主线程在 NPU 跑当前 batch 期间可并行准备下一批的 processor 打包。
    total = len(my_queue)
    save_step = max(BATCH_SIZE * 2, 10)
    pbar = tqdm(total=total, desc=f"Rank {rank}", position=rank)

    last_saved = len(results)
    pending_batch = []

    def maybe_save():
        nonlocal last_saved
        if len(results) - last_saved >= save_step:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            last_saved = len(results)

    # NPU worker：从 npu_inbox 取 (batch, cpu_inputs)，跑推理，结果回 npu_outbox
    # maxsize=2 -> 主线程最多比 NPU 提前 1 个 batch (1 个在 NPU 跑 + 1 个在队列)
    npu_inbox = _queue.Queue(maxsize=2)
    npu_outbox = _queue.Queue()

    def npu_worker():
        while True:
            item = npu_inbox.get()
            if item is None:
                return
            batch, cpu_inputs = item
            try:
                br = inferencer.run_inference(batch, cpu_inputs)
            except BaseException as e:
                # 用 traceback 打全栈到 stderr，避免 NPU 错误信息为空时啥也看不到
                tb = traceback.format_exc()
                err_summary = f"{type(e).__name__}: {str(e) or repr(e) or '<empty msg>'}"
                print(f"\n[NPU worker @ {device}] EXCEPTION:\n{tb}", flush=True)
                # 同时 dump 一些诊断信息，方便定位是哪个 sample / 什么 input 触发的
                try:
                    shapes = {}
                    for k, v in cpu_inputs.items():
                        if hasattr(v, "shape"):
                            shapes[k] = tuple(v.shape)
                        elif isinstance(v, list):
                            shapes[k] = f"list[{len(v)}]"
                    print(f"[NPU worker @ {device}] failing batch shapes: {shapes}", flush=True)
                    print(f"[NPU worker @ {device}] failing batch ids: {[x['video_path'].rsplit('/', 1)[-1] for x in batch]}", flush=True)
                except Exception:
                    pass
                br = [{"id": x["video_path"], "answers": [f"NPU Error: {err_summary}"]} for x in batch]
            npu_outbox.put(br)

    npu_thread = threading.Thread(target=npu_worker, daemon=True)
    npu_thread.start()

    submitted = 0  # 已提交给 NPU 的 batch 数
    consumed = 0   # 已从 NPU 取回结果的 batch 数

    def drain_npu_results(block_until=None):
        """从 npu_outbox 取已完成的 batch 结果写入 results。
        block_until=None 表示非阻塞；否则阻塞直到 consumed 达到 block_until。"""
        nonlocal consumed
        while True:
            if block_until is None:
                try:
                    br = npu_outbox.get_nowait()
                except _queue.Empty:
                    return
            else:
                if consumed >= block_until:
                    return
                br = npu_outbox.get()
            results.extend(br)
            pbar.update(len(br))
            consumed += 1
            maybe_save()

    # 诊断计时 (仅 rank 0)
    t_loop_start = time.perf_counter() if rank == 0 else None
    t_build_total = 0.0
    t_wait_npu_total = 0.0
    build_stats = {} if rank == 0 else None  # 细分 build_inputs_cpu 的耗时来源
    diag_every = 5

    if total > 0:
        with ThreadPoolExecutor(max_workers=PREPROCESS_WORKERS) as executor:
            future_to_path = {}
            for messages, video_path in my_queue:
                if not os.path.exists(video_path):
                    results.append({"id": video_path, "answers": ["Error: Video not found"]})
                    pbar.update(1)
                    continue
                fut = executor.submit(preprocess_sample, processor, messages, video_path)
                future_to_path[fut] = video_path

            remaining = len(future_to_path)
            for fut in as_completed(future_to_path):
                video_path = future_to_path[fut]
                remaining -= 1
                try:
                    item = fut.result()
                except Exception as e:
                    results.append({"id": video_path, "answers": [f"Preprocess Critical Error: {str(e)}"]})
                    pbar.update(1)
                    maybe_save()
                    continue

                if item.get("error"):
                    results.append({"id": item["video_path"], "answers": [f"Error: {item['error']}"]})
                    pbar.update(1)
                    maybe_save()
                else:
                    pending_batch.append(item)

                should_infer = (
                    len(pending_batch) >= BATCH_SIZE
                    or (remaining == 0 and len(pending_batch) > 0)
                )
                if should_infer:
                    # Stage 2: 主线程 CPU 工作 —— processor 打包，与 NPU 当前 batch 并行
                    if rank == 0:
                        _t0 = time.perf_counter()
                    try:
                        cpu_inputs = inferencer.build_inputs_cpu(pending_batch, _stats=build_stats)
                    except Exception as e:
                        err = f"build_inputs error: {str(e)}"
                        print(f"[Rank {rank}] {err}", flush=True)
                        results.extend([{"id": x["video_path"], "answers": [err]} for x in pending_batch])
                        pbar.update(len(pending_batch))
                        pending_batch = []
                        maybe_save()
                        continue
                    if rank == 0:
                        t_build_total += time.perf_counter() - _t0

                    # Stage 3: 推给 NPU worker；队列满时阻塞 (NPU 跟不上时自然限流)
                    if rank == 0:
                        _t0 = time.perf_counter()
                    npu_inbox.put((pending_batch, cpu_inputs))
                    if rank == 0:
                        t_wait_npu_total += time.perf_counter() - _t0

                    submitted += 1
                    pending_batch = []

                    # 非阻塞地把已完成的 batch 取回来
                    drain_npu_results(block_until=None)

                    if rank == 0 and submitted % diag_every == 0:
                        elapsed = time.perf_counter() - t_loop_start
                        sps = submitted * BATCH_SIZE / elapsed
                        b_collate = build_stats.get("collate", 0.0)
                        print(
                            f"\n[Rank 0 timing] submitted={submitted} consumed={consumed} "
                            f"elapsed={elapsed:.1f}s | collate={b_collate:.1f}s "
                            f"({100*b_collate/elapsed:.0f}%) | "
                            f"wait_for_NPU={t_wait_npu_total:.1f}s "
                            f"({100*t_wait_npu_total/elapsed:.0f}%) | "
                            f"throughput={sps:.2f} samples/s",
                            flush=True,
                        )

    # 所有 batch 都已提交；告诉 NPU 线程结束并把剩余结果取回
    drain_npu_results(block_until=submitted)
    npu_inbox.put(None)
    npu_thread.join(timeout=120)

    pbar.close()

    # 兜底：极少数情况下 pending_batch 可能因异常残留，同步推理收尾
    if pending_batch:
        batch_results = inferencer.infer_batch(pending_batch)
        results.extend(batch_results)
        pending_batch = []

    # 最终保存
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Rank {rank}: 本地推理全部完成。")


def merge_results(final_output_path, world_size):
    """
    采用字典根据 ID 去重，并且如果最终大文件已存在，先将其读入，防止覆盖历史数据。
    """
    all_results_dict = {}

    # 读入已有的全局最终结果 (断点续传保留)
    if os.path.exists(final_output_path):
        try:
            with open(final_output_path, 'r', encoding='utf-8') as f:
                for item in json.load(f):
                    all_results_dict[item['id']] = item
        except Exception as e:
            print(f"读取历史汇总文件警告: {e}")

    # 合并各个 Rank 的临时文件
    base_name = final_output_path.rsplit('.', 1)[0]
    for r in range(world_size):
        temp_file = f"{base_name}_rank{r}.json"
        if os.path.exists(temp_file):
            try:
                with open(temp_file, 'r', encoding='utf-8') as f:
                    for item in json.load(f):
                        all_results_dict[item['id']] = item
                os.remove(temp_file)
            except Exception as e:
                print(f"合并 {temp_file} 失败: {e}")

    final_list = list(all_results_dict.values())
    with open(final_output_path, 'w', encoding='utf-8') as f:
        json.dump(final_list, f, ensure_ascii=False, indent=2)
    print(f"合并彻底完成，输出文件共包含 {len(final_list)} 条唯一数据！")


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = int(os.getenv("LOCAL_RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))

    os.environ["HCCL_CONNECT_TIMEOUT"] = "7200"

    if not dist.is_initialized():
        dist.init_process_group(backend="hccl", world_size=world_size, rank=local_rank)

    main_worker(local_rank, world_size, model_args, data_args, training_args)

    if dist.is_initialized():
        # barrier：保证所有 rank 写完文件后再 destroy + merge
        dist.barrier()
        dist.destroy_process_group()

    if local_rank == 0:
        merge_results(data_args.output_json, world_size)


if __name__ == "__main__":
    main()
