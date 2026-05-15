import os
import json
import threading
import heapq
import time
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
    # 强烈建议使用 AutoModelForImageTextToText 作为兜底，它可以自动路由 Dense 或 MoE 模型架构
    from transformers import AutoModelForImageTextToText as TargetVLModel, AutoProcessor, HfArgumentParser

from infer_params import DataArguments, ModelArguments, TrainingArguments
from qwen_vl_utils import process_vision_info

# ====================== 配置参数 ======================
# CPU 预处理线程数 (视频解码 + chat_template + process_vision_info)
# 与 NPU 推理流水线并行，不直接吃显存；过大反而抢 CPU/RAM
PREPROCESS_WORKERS = int(os.environ.get("PREPROCESS_WORKERS", 6))

# NPU 单次推理的批大小。显存是这里的主要瓶颈：
#   - Qwen3.5-VL-8B 权重 ~16GB
#   - thinking 模式 max_new_tokens=4096，KV cache 与 batch 线性
#   - 910B 单卡 65GB，batch=4 通常稳，batch=6/8 视实际显存调
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 4))

torch.npu.config.allow_internal_format = False
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"


# ====================== CPU 预处理 ======================
def preprocess_sample(processor, messages, video_path):
    """单样本 CPU 预处理。返回打包好的 dict，供后续 batch 拼接使用。"""
    try:
        # 强制覆盖视频参数 (与原代码保持一致)
        for msg in messages:
            if msg.get("role") == "user":
                for content_item in msg.get("content", []):
                    if content_item.get("type") == "video":
                        content_item["fps"] = 8.0
                        content_item["max_pixels"] = 589824  # 602112

        # 应用聊天模板 (思考模式)
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

        # 视觉处理 (Qwen3 特有逻辑)
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            return_video_kwargs=True,
            image_patch_size=16,
            return_video_metadata=True,
        )

        # Qwen3 视频元数据解包逻辑
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


# ====================== NPU 批量推理 ======================
class BatchedInferencer:
    """一次接收 N 个预处理好的样本，拼成 batch 喂给 NPU.generate。"""

    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device

        # 关键：autoregressive generate 必须 left padding
        # 否则 batch 中较短样本右侧的 pad token 会让 decode 阶段在 pad 上预测
        self.processor.tokenizer.padding_side = "left"

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.processor.tokenizer.eos_token_id

        # Qwen3.5 开启思考过程的生成参数 (与原单样本版本一致)
        self.gen_kwargs = {
            "max_new_tokens": 4096,
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "repetition_penalty": 1.05,
            "pad_token_id": pad_id,
        }

    def build_inputs_cpu(self, batch):
        """主线程 CPU 工作：合并 vision 输入 + processor 打包 + padding。返回 CPU 端 BatchFeature。
        不做 .to(device)，让 NPU 传输/计算都在 NPU 工作线程内同一 stream 上。"""
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
                merged_kwargs[k] = vals[0]

        return self.processor(
            text=texts,
            images=all_images,
            videos=all_videos,
            video_metadata=all_video_metadatas,
            padding=True,
            return_tensors="pt",
            **merged_kwargs,
        )

    def run_inference(self, batch, cpu_inputs):
        """NPU 工作线程调用：.to(device) → generate → decode → 拼结果"""
        inputs = cpu_inputs.to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **self.gen_kwargs)
        prompt_len = inputs.input_ids.shape[1]
        generated_ids_trimmed = generated_ids[:, prompt_len:]
        out_texts = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        # 去掉 torch.npu.empty_cache() —— 每 batch 调一次会插强同步点拖慢节奏
        return [
            {"id": item["video_path"], "answers": [out_texts[i]]}
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
    inferencer = BatchedInferencer(model, processor, device)

    # 准备输出文件
    output_file = f"{data_args.output_json.rsplit('.', 1)[0]}_rank{rank}.json"

    # 读取已完成结果
    completed_ids = set()

    # 本 Rank 临时文件
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing = json.load(f)
                completed_ids.update({r['id'] for r in existing})
        except: pass

    # 全局合并文件
    if os.path.exists(data_args.output_json):
        try:
            with open(data_args.output_json, 'r', encoding='utf-8') as f:
                existing_global = json.load(f)
                completed_ids.update({r['id'] for r in existing_global})
        except: pass

    results = []
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            results = json.load(f)

    # 任务分配 (仅 Rank 0 读取并分发)
    if rank == 0:
        with open(data_args.data_path, 'r', encoding='utf-8') as f:
            test_samples = json.load(f)

        valid_samples = []
        for sample in test_samples:
            try:
                first_msg = sample[0] if isinstance(sample, list) else sample
                content = first_msg.get("content", [])
                video_path = next((item["video"] for item in content if item["type"] == "video"), None)
                if video_path and video_path not in completed_ids:
                    valid_samples.append((sample, video_path))
            except: continue

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

        # ★ 每个 rank 内部按文件大小降序排序：同一 batch 内 token 数差异变小，
        # left-padding 浪费的 prefill 算力随之减少。
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

    # 过滤掉本Rank已经跑过的 (Double check)
    my_queue = [t for t in my_tasks if t[1] not in completed_ids]

    print(f"Rank {rank}: 分配到 {len(my_queue)} 个任务，预处理并行数 {PREPROCESS_WORKERS}，推理 batch={BATCH_SIZE}")

    # ========== 三段流水线 ==========
    # Stage 1 (CPU 线程池): 视频解码 + chat_template + vision tokenize
    # Stage 2 (主线程):     凑齐 BATCH_SIZE → build_inputs_cpu (processor 打包)
    # Stage 3 (NPU 单线程):  .to(device) → model.generate → decode
    # Stage 2 / Stage 3 通过队列异步衔接，主线程在 NPU 跑当前 batch 时可并行准备下一批
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
            except Exception as e:
                err = f"NPU Error: {str(e)}"
                print(f"[NPU worker @ {device}] {err}", flush=True)
                br = [{"id": x["video_path"], "answers": [err]} for x in batch]
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

    # 诊断 (rank 0 only)
    t_loop_start = time.perf_counter() if rank == 0 else None
    t_build_total = 0.0
    t_wait_npu_total = 0.0
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
                        cpu_inputs = inferencer.build_inputs_cpu(pending_batch)
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

                    if rank == 0:
                        _t0 = time.perf_counter()
                    npu_inbox.put((pending_batch, cpu_inputs))  # 队列满则阻塞 (背压)
                    if rank == 0:
                        t_wait_npu_total += time.perf_counter() - _t0

                    submitted += 1
                    pending_batch = []
                    drain_npu_results(block_until=None)

                    if rank == 0 and submitted % diag_every == 0:
                        elapsed = time.perf_counter() - t_loop_start
                        sps = submitted * BATCH_SIZE / elapsed
                        print(
                            f"\n[Rank 0 timing] submitted={submitted} consumed={consumed} "
                            f"elapsed={elapsed:.1f}s | build_inputs={t_build_total:.1f}s "
                            f"({100*t_build_total/elapsed:.0f}%) | wait_for_NPU={t_wait_npu_total:.1f}s "
                            f"({100*t_wait_npu_total/elapsed:.0f}%) | throughput={sps:.2f} samples/s",
                            flush=True,
                        )

    # 所有 batch 都已提交；把剩余结果取回，然后告诉 NPU 线程结束
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
    print(f"Rank {rank}: 全部推断完成，正在写入 done 标记。")

    # ================= 写入完成标记 =================
    done_flag_file = f"{output_file}.done"
    with open(done_flag_file, 'w') as f:
        f.write("Finished")


def merge_results(final_output_path, world_size):
    all_results = []

    # 先读取已存在的全局 result 文件，防止覆盖之前跑完的数据
    if os.path.exists(final_output_path):
        try:
            with open(final_output_path, 'r', encoding='utf-8') as f:
                all_results.extend(json.load(f))
        except: pass

    base_name = final_output_path.rsplit('.', 1)[0]
    for r in range(world_size):
        temp_file = f"{base_name}_rank{r}.json"
        done_file = f"{base_name}_rank{r}.json.done"

        # 读取并合入临时文件数据
        if os.path.exists(temp_file):
            try:
                with open(temp_file, 'r', encoding='utf-8') as f:
                    all_results.extend(json.load(f))
                os.remove(temp_file)
            except Exception as e:
                print(f"合并临时文件 {temp_file} 失败: {e}")

        # 清理 done 标记文件
        if os.path.exists(done_file):
            try:
                os.remove(done_file)
            except: pass

    # 基于视频 id 进行字典去重，防止多次中断重启导致数据重复
    unique_results = list({item['id']: item for item in all_results}.values())

    with open(final_output_path, 'w', encoding='utf-8') as f:
        json.dump(unique_results, f, ensure_ascii=False, indent=2)
    print(f"合并彻底完成，最终共 {len(unique_results)} 条唯一数据")


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = int(os.getenv("LOCAL_RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))

    os.environ["HCCL_CONNECT_TIMEOUT"] = "7200"

    if not dist.is_initialized():
        dist.init_process_group(backend="hccl", world_size=world_size, rank=local_rank)

    main_worker(local_rank, world_size, model_args, data_args, training_args)

    # 强制清空 NPU 缓存并同步，防止硬件层异步卡死
    torch.npu.empty_cache()
    torch.npu.synchronize()

    # ================= 基于 done 标记的安全等待机制 =================
    if local_rank == 0:
        print("\nRank 0: 正在等待所有节点彻底执行完成...")
        base_name = data_args.output_json.rsplit('.', 1)[0]

        wait_time = 0
        max_wait_time = 3600 * 12  # 12 小时上限

        while wait_time < max_wait_time:
            all_done = True
            for r in range(world_size):
                done_file = f"{base_name}_rank{r}.json.done"
                if not os.path.exists(done_file):
                    all_done = False
                    break

            if all_done:
                print("\nRank 0: 检测到所有节点的 .done 标记，开始最终合并...")
                break

            time.sleep(15)
            wait_time += 15

        if wait_time >= max_wait_time:
            print("\nRank 0: 警告！等待时间超时，强制开始合并现有数据！")

        merge_results(data_args.output_json, world_size)


if __name__ == "__main__":
    main()
