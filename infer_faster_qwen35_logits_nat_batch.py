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
                        content_item["max_pixels"] = 602112

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

        return {
            "video_path": video_path,
            "text": text,
            "image_inputs": image_inputs,
            "video_inputs": video_inputs,
            "video_metadatas": video_metadatas,
            "video_kwargs": video_kwargs or {},
            "error": None,
        }
    except Exception as e:
        return {"video_path": video_path, "error": str(e)}


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

    def infer_batch(self, batch):
        """batch: List[preprocessed dict]。返回 List[{id, answers, logits}]"""
        if not batch:
            return []

        try:
            texts = [item["text"] for item in batch]

            # 跨样本拼接 vision 输入
            all_images = []
            for item in batch:
                if item["image_inputs"]:
                    all_images.extend(item["image_inputs"])
            all_images = all_images or None

            all_videos = []
            for item in batch:
                if item["video_inputs"]:
                    all_videos.extend(item["video_inputs"])
            all_videos = all_videos or None

            all_video_metadatas = []
            for item in batch:
                if item["video_metadatas"]:
                    all_video_metadatas.extend(item["video_metadatas"])
            all_video_metadatas = all_video_metadatas or None

            # 合并 video_kwargs：
            #   - list/tuple 值是 per-video 的 (例如 fps)，跨样本拼平
            #   - 标量值是全局开关 (例如 do_sample_frames=False)，所有样本本来就一致，
            #     原样透传一个标量；强行包成 list 会被 processor 校验器拒绝
            merged_kwargs = {}
            all_kw_keys = set()
            for item in batch:
                if item["video_kwargs"]:
                    all_kw_keys.update(item["video_kwargs"].keys())
            for k in all_kw_keys:
                vals = [item["video_kwargs"][k] for item in batch
                        if item["video_kwargs"] and k in item["video_kwargs"]]
                if not vals:
                    continue
                if isinstance(vals[0], (list, tuple)):
                    flat = []
                    for v in vals:
                        flat.extend(v)
                    merged_kwargs[k] = flat
                else:
                    # 标量：同一次运行里所有样本应一致，取第一个透传
                    merged_kwargs[k] = vals[0]

            inputs = self.processor(
                text=texts,
                images=all_images,
                videos=all_videos,
                video_metadata=all_video_metadatas,
                padding=True,
                return_tensors="pt",
                **merged_kwargs,
            ).to(self.device)

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

            del inputs, outputs, first_step_logits, generated_ids_trimmed
            torch.npu.empty_cache()

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
        except Exception as e:
            # 整批失败：所有样本都标 Error，避免静默丢任务
            err = f"Batch Error: {str(e)}"
            print(f"[Batch fail @ {self.device}] {err}")
            try:
                torch.npu.empty_cache()
            except: pass
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
        task_lists = [[] for _ in range(world_size)]

        for size, sample, video_path in sample_sizes:
            load, w = heapq.heappop(worker_loads)
            task_lists[w].append((sample, video_path))
            heapq.heappush(worker_loads, (load + size, w))

        dist_obj = [task_lists]
    else:
        dist_obj = [None]

    # 广播任务
    dist.broadcast_object_list(dist_obj, src=0)
    my_tasks = dist_obj[0][rank]

    # 过滤掉本 Rank 已经跑过的 (Double check)
    my_queue = [t for t in my_tasks if t[1] not in completed_ids]

    print(f"Rank {rank}: 分配到 {len(my_queue)} 个任务，预处理并行数 {PREPROCESS_WORKERS}，推理 batch={BATCH_SIZE}")

    # ========== 流水线：CPU 预处理并行 → 主线程凑批 → NPU 批量推理 ==========
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
                    # 预处理失败：单条记 Error，不进 batch
                    results.append({"id": item["video_path"], "answers": [f"Error: {item['error']}"]})
                    pbar.update(1)
                    maybe_save()
                else:
                    pending_batch.append(item)

                # 触发推理：batch 满 / 没有后续待预处理样本
                should_infer = (
                    len(pending_batch) >= BATCH_SIZE
                    or (remaining == 0 and len(pending_batch) > 0)
                )
                if should_infer:
                    batch_results = inferencer.infer_batch(pending_batch)
                    results.extend(batch_results)
                    pbar.update(len(batch_results))
                    pending_batch = []
                    maybe_save()

    pbar.close()

    # 兜底：还有零散没推的再 flush 一次 (理论上 remaining==0 分支已处理)
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
