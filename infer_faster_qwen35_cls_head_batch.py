"""
Qwen3.5-VL cls_head 推理的批量版本。

基于 infer_faster_qwen35_cls_head.py：
  - 用 model.forward(output_hidden_states=True) 取最后一层最后一个 token 的 hidden state
  - 过 BinaryClassificationHead 得到 [neg, pos] 两类 logits
  - 输出格式与原版一致：{"id", "answers": [pred_text], "logits": {neg_label, pos_label}}

改造点（与 infer_faster_qwen35_logits_nat_batch.py 完全相同的流水线）：
  - 单样本 processor() 调用挪进 PREPROCESS_WORKERS 并行做
  - 主线程只做轻量 collate（左 pad + cat）
  - NPU 推理放独立线程，与主线程的 collate 异步重叠
  - rank 0 打印诊断行

★ 关键正确性：tokenizer.padding_side = "left" 后，batch 里每个样本的
  「最后一个真实 token」都恰好落在序列末尾 (index -1)，所以
  last_hidden[:, -1, :] 对整个 batch 一次取出全部样本的分类特征，
  与原单样本版本逐样本取 [:, -1, :] 完全等价。

Required files in --model_id 目录：
    - 标准 merged model (config.json, model.safetensors, tokenizer ...)
    - cls_head.bin            (BinaryClassificationHead 的 state_dict)
    - cls_head_meta.json      ({"label_map": {"0": "<neg>", "1": "<pos>"}, "hidden_size": ...})
"""

import os
import json
import heapq
import time
import threading
import traceback
import queue as _queue
from pathlib import Path
from tqdm import tqdm
import torch
import torch_npu
import torch.distributed as dist
from concurrent.futures import ThreadPoolExecutor, as_completed

# [Qwen3.5 通用引入]
try:
    from transformers import Qwen3_5VLForConditionalGeneration as TargetVLModel, AutoProcessor, HfArgumentParser
except ImportError:
    from transformers import AutoModelForImageTextToText as TargetVLModel, AutoProcessor, HfArgumentParser

from infer_params import DataArguments, ModelArguments, TrainingArguments
from qwen_vl_utils import process_vision_info

# Import the head class definition. Adjust path if your LlamaFactory checkout is elsewhere.
import sys
sys.path.insert(0, "/home/ma-user/work/lyf/LlamaFactory-qwen35/src")
from llamafactory.model.cls_head import BinaryClassificationHead  # noqa: E402

# ====================== 配置 ======================
# CPU 预处理并行线程数 (视频解码 + chat_template + 单样本 processor)
PREPROCESS_WORKERS = int(os.environ.get("PREPROCESS_WORKERS", 6))

# NPU 单次批大小。cls_head 模式只做一次 forward (无 KV cache 增长)，显存压力小。
# 910B 65GB 单卡推荐 8 起步，富裕时上调到 12/16。
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 8))

# 视频参数 (与原 cls_head 版一致，可用环境变量覆盖；默认 fps=8.0, max_pixels=602112)
VIDEO_FPS = float(os.getenv("INFER_VIDEO_FPS", "8.0"))
VIDEO_MAX_PIXELS = int(os.getenv("INFER_VIDEO_MAX_PIXELS", "602112"))

torch.npu.config.allow_internal_format = False
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"


# ====================== CPU 预处理 ======================
def preprocess_sample(processor, messages, video_path):
    """单样本 CPU 预处理，与原 cls_head 单样本版本严格对齐。"""
    try:
        for msg in messages:
            if msg.get("role") == "user":
                for content_item in msg.get("content", []):
                    if content_item.get("type") == "video":
                        content_item["fps"] = VIDEO_FPS
                        content_item["max_pixels"] = VIDEO_MAX_PIXELS

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        # 注意：cls_head 版本不 strip 空 <think> 块 —— 与原 cls_head 单样本脚本保持一致。

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
        # 跨 8 ranks 并发时单样本 processor 远快于主线程 batched processor。
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


# ====================== Custom collator ======================
def _collate_qwen_vl(per_sample_inputs, pad_token_id):
    """把 N 份 batch=1 的 BatchFeature 拼成 batch=N。

    Qwen3.5-VL processor 输出 5 个字段：
      - input_ids / attention_mask / mm_token_type_ids: shape [1, seq_len_i] —— 左 pad
      - pixel_values_videos: shape [num_patches_i, 1536] —— 沿 dim 0 cat
      - video_grid_thw: shape [num_videos_i, 3] —— 沿 dim 0 cat

    通用规则: 所有 shape=[1, X] 且 X 跨样本不一致的 tensor 都 left-pad；
    其它 tensor 沿 dim 0 cat；list/标量按列表 extend。
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
            # 是否 pad 只看「本字段 dim-1 各样本是否一致」——
            # video_grid_thw 是 [1, 3] (3 不是 seq_len)，各样本一致 → 不 pad，直接 cat。
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
                out[k] = torch.cat(vals, dim=0)
        elif torch.is_tensor(v0):
            try:
                out[k] = torch.cat(vals, dim=0)
            except Exception:
                out[k] = vals
        elif isinstance(v0, list):
            merged = []
            for v in vals:
                merged.extend(v)
            out[k] = merged
        else:
            out[k] = vals

    return BatchFeature(data=out)


# ====================== NPU 批量推理 (forward + cls_head) ======================
class BatchedClsHeadInferencer:
    def __init__(self, model, processor, cls_head, label_map, device):
        self.model = model
        self.processor = processor
        self.cls_head = cls_head
        self.cls_head_dtype = next(cls_head.parameters()).dtype
        self.label_map = label_map           # {"0": <neg_text>, "1": <pos_text>}
        self.neg_label = label_map["0"]
        self.pos_label = label_map["1"]
        self.device = device

        # ★ left padding：让 batch 里每个样本的最后一个真实 token 都落在 index -1，
        #   这样 last_hidden[:, -1, :] 一次性取出整 batch 的分类特征。
        self.processor.tokenizer.padding_side = "left"
        self.pad_id = self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id

    def build_inputs_cpu(self, batch, _stats=None):
        """主线程 CPU 工作：把 N 份 batch=1 的 per-sample BatchFeature collate 成 batch=N。"""
        if _stats is not None:
            _t0 = time.perf_counter()

        per_sample = [item["cpu_inputs"] for item in batch]
        inputs = _collate_qwen_vl(per_sample, pad_token_id=self.pad_id)

        if _stats is not None:
            _stats["collate"] = _stats.get("collate", 0.0) + (time.perf_counter() - _t0)
        return inputs

    def run_inference(self, batch, cpu_inputs):
        """NPU 工作线程调用：.to(device) → forward → cls_head → 拼结果。"""
        inputs = cpu_inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )

        # 最后一层 hidden states: [B, T, D]
        last_hidden = outputs.hidden_states[-1]
        # left padding 下，每个样本最后一个真实 token 都在序列末尾 → [:, -1, :] 即分类特征
        h = last_hidden[:, -1, :]                                  # [B, D]
        cls_logits = self.cls_head(h.to(self.cls_head_dtype))      # [B, 2]

        # 一次性搬到 CPU，避免逐样本 .item() 同步开销
        cls_logits_list = cls_logits.float().cpu().tolist()        # [B, 2]

        results = []
        for i, item in enumerate(batch):
            logit_neg = cls_logits_list[i][0]   # class 0 = negative
            logit_pos = cls_logits_list[i][1]   # class 1 = positive
            pred_text = self.pos_label if logit_pos > logit_neg else self.neg_label
            results.append({
                "id": item["video_path"],
                "answers": [pred_text],
                "logits": {self.neg_label: logit_neg, self.pos_label: logit_pos},
            })
        return results

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
    device = f"npu:{rank}"
    torch.npu.set_device(device)

    print(f"Rank {rank}: 加载 merged Qwen3.5 模型 (BATCH_SIZE={BATCH_SIZE}, PREPROC_WORKERS={PREPROCESS_WORKERS})...")
    model = TargetVLModel.from_pretrained(
        model_args.model_id,
        torch_dtype="auto",
        device_map=None,
        attn_implementation="flash_attention_2",
    ).eval().to(device)

    processor = AutoProcessor.from_pretrained(model_args.model_id)

    # ---- 加载分类头 ----
    head_bin = Path(model_args.model_id) / "cls_head.bin"
    head_meta = Path(model_args.model_id) / "cls_head_meta.json"
    if not head_bin.exists() or not head_meta.exists():
        raise FileNotFoundError(
            f"cls_head.bin / cls_head_meta.json not found in {model_args.model_id}. "
            "Make sure your merge step copied them from the checkpoint dir."
        )
    with open(head_meta, encoding="utf-8") as f:
        meta = json.load(f)
    if "label_map" not in meta:
        meta["label_map"] = {
            "0": meta.get("negative_text", meta.get("safe_text", "class_0")),
            "1": meta.get("positive_text", meta.get("high_risk_text", "class_1")),
        }
    cls_head = BinaryClassificationHead(meta["hidden_size"], dropout=meta.get("dropout", 0.0))
    cls_head.load_state_dict(torch.load(head_bin, map_location="cpu"))
    cls_head = cls_head.to(device).to(torch.bfloat16).eval()
    print(
        f"Rank {rank}: loaded cls_head (hidden={meta['hidden_size']}); "
        f"label_map: 0='{meta['label_map']['0']}', 1='{meta['label_map']['1']}'"
    )

    inferencer = BatchedClsHeadInferencer(model, processor, cls_head, meta["label_map"], device)

    # 准备输出文件
    output_file = f"{data_args.output_json.rsplit('.', 1)[0]}_rank{rank}.json"

    # 读取本 Rank 已完成结果
    completed_ids = set()
    results = []
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                results = json.load(f)
                completed_ids = {r["id"] for r in results}
        except Exception:
            pass

    # 数据分配 (仅 Rank 0 读取并分发)
    if rank == 0:
        global_completed_ids = set()
        if os.path.exists(data_args.output_json):
            try:
                with open(data_args.output_json, "r", encoding="utf-8") as f:
                    for r in json.load(f):
                        global_completed_ids.add(r["id"])
            except Exception:
                pass

        with open(data_args.data_path, "r", encoding="utf-8") as f:
            test_samples = json.load(f)

        valid_samples = []
        for sample in test_samples:
            try:
                first_msg = sample[0] if isinstance(sample, list) else sample
                content = first_msg.get("content", [])
                video_path = next((c["video"] for c in content if c["type"] == "video"), None)
                if video_path and video_path not in global_completed_ids and video_path not in completed_ids:
                    valid_samples.append((sample, video_path))
            except Exception:
                continue

        sample_sizes = []
        for sample, video_path in valid_samples:
            size = os.path.getsize(video_path) if os.path.exists(video_path) else 0
            sample_sizes.append((size, sample, video_path))
        sample_sizes.sort(key=lambda x: x[0], reverse=True)

        worker_loads = [(0, w) for w in range(world_size)]
        heapq.heapify(worker_loads)
        task_lists_sized = [[] for _ in range(world_size)]
        for size, sample, video_path in sample_sizes:
            load, w = heapq.heappop(worker_loads)
            task_lists_sized[w].append((size, sample, video_path))
            heapq.heappush(worker_loads, (load + size, w))

        # ★ rank 内按文件大小降序排：相邻 batch token 数相近，减少 left-pad 浪费
        task_lists = []
        for w in range(world_size):
            task_lists_sized[w].sort(key=lambda x: x[0], reverse=True)
            task_lists.append([(s, v) for _, s, v in task_lists_sized[w]])

        dist_obj = [task_lists]
    else:
        dist_obj = [None]

    dist.broadcast_object_list(dist_obj, src=0)
    my_tasks = dist_obj[0][rank]
    my_queue = [t for t in my_tasks if t[1] not in completed_ids]
    print(f"Rank {rank}: 分配到 {len(my_queue)} 个任务，预处理并行数 {PREPROCESS_WORKERS}，推理 batch={BATCH_SIZE}")

    # ========== 三段流水线 ==========
    # Stage 1 (CPU 线程池): 视频解码 + chat_template + 单样本 processor
    # Stage 2 (主线程):     凑齐 BATCH_SIZE → collate
    # Stage 3 (NPU 单线程):  .to(device) → model.forward → cls_head
    total = len(my_queue)
    save_step = max(BATCH_SIZE * 2, 10)
    pbar = tqdm(total=total, desc=f"Rank {rank}", position=rank)

    last_saved = len(results)
    pending_batch = []

    def maybe_save():
        nonlocal last_saved
        if len(results) - last_saved >= save_step:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            last_saved = len(results)

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
                tb = traceback.format_exc()
                err_summary = f"{type(e).__name__}: {str(e) or repr(e) or '<empty msg>'}"
                print(f"\n[NPU worker @ {device}] EXCEPTION:\n{tb}", flush=True)
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

    submitted = 0
    consumed = 0

    def drain_npu_results(block_until=None):
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
    t_wait_npu_total = 0.0
    build_stats = {} if rank == 0 else None
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
                        _t0 = time.perf_counter()
                    npu_inbox.put((pending_batch, cpu_inputs))
                    if rank == 0:
                        t_wait_npu_total += time.perf_counter() - _t0

                    submitted += 1
                    pending_batch = []

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

    drain_npu_results(block_until=submitted)
    npu_inbox.put(None)
    npu_thread.join(timeout=120)

    pbar.close()

    if pending_batch:
        batch_results = inferencer.infer_batch(pending_batch)
        results.extend(batch_results)
        pending_batch = []

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Rank {rank}: 本地推理全部完成。")


def merge_results(final_output_path, world_size):
    all_results_dict = {}
    if os.path.exists(final_output_path):
        try:
            with open(final_output_path, "r", encoding="utf-8") as f:
                for item in json.load(f):
                    all_results_dict[item["id"]] = item
        except Exception as e:
            print(f"读取历史汇总文件警告: {e}")

    base_name = final_output_path.rsplit(".", 1)[0]
    for r in range(world_size):
        temp_file = f"{base_name}_rank{r}.json"
        if os.path.exists(temp_file):
            try:
                with open(temp_file, "r", encoding="utf-8") as f:
                    for item in json.load(f):
                        all_results_dict[item["id"]] = item
                os.remove(temp_file)
            except Exception as e:
                print(f"合并 {temp_file} 失败: {e}")

    final_list = list(all_results_dict.values())
    with open(final_output_path, "w", encoding="utf-8") as f:
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
        dist.barrier()
        dist.destroy_process_group()

    if local_rank == 0:
        merge_results(data_args.output_json, world_size)


if __name__ == "__main__":
    main()
