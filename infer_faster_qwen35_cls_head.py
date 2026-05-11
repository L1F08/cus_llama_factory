"""Distributed inference for cls_head-trained VLM (NPU + ThreadPoolExecutor).

Mirrors `infer_faster_qwen35_logits_nat.py` but replaces `model.generate()` with
`model.forward()` + classification head — produces the SAME output format
(`logits.{neg_label, pos_label}`) so the existing eval script works unchanged.

Required files in --merged_dir:
    - Standard merged model (config.json, model.safetensors, tokenizer files, ...)
    - cls_head.bin              (state_dict of BinaryClassificationHead)
    - cls_head_meta.json        ({"label_map": {"0": "<neg>", "1": "<pos>"}, ...})
"""

import os
import json
import threading
import heapq
from pathlib import Path
from tqdm import tqdm
import torch
import torch_npu
import torch.distributed as dist
from concurrent.futures import ThreadPoolExecutor, as_completed

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


# ====================== 配置参数 ======================
PARALLEL_WORKERS = 6

torch.npu.config.allow_internal_format = False
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"


# ====================== 模型池 ======================
class ModelPool:
    """Threaded inference pool — CPU vision-info parallel + locked NPU forward."""

    def __init__(self, model, processor, cls_head, label_map, device,
                 video_fps=8.0, video_max_pixels=602112):
        self.model = model
        self.processor = processor
        self.cls_head = cls_head
        self.cls_head_dtype = next(cls_head.parameters()).dtype
        self.label_map = label_map  # {"0": <neg_text>, "1": <pos_text>}
        self.neg_label = label_map["0"]
        self.pos_label = label_map["1"]
        self.device = device
        self.lock = threading.Lock()
        self.video_fps = video_fps
        self.video_max_pixels = video_max_pixels

    def infer_single(self, messages, video_path):
        try:
            # 1. CPU stage: preprocessing (lock-free)
            for msg in messages:
                if msg.get("role") == "user":
                    for content_item in msg.get("content", []):
                        if content_item.get("type") == "video":
                            content_item["fps"] = self.video_fps
                            content_item["max_pixels"] = self.video_max_pixels

            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

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

            # 2. NPU stage: forward + cls_head (locked)
            with self.lock:
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    video_metadata=video_metadatas,
                    padding=True,
                    return_tensors="pt",
                    **video_kwargs,
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(
                        **inputs,
                        output_hidden_states=True,
                        return_dict=True,
                        use_cache=False,
                    )

                # Last hidden state at the last position (right before generation)
                last_hidden = outputs.hidden_states[-1]            # [1, T, D]
                h = last_hidden[:, -1, :]                          # [1, D]
                cls_logits = self.cls_head(h.to(self.cls_head_dtype))  # [1, 2]

                # Class 0 = negative, Class 1 = positive
                logit_neg = float(cls_logits[0, 0].item())
                logit_pos = float(cls_logits[0, 1].item())

                del inputs, outputs, last_hidden, cls_logits
                torch.npu.empty_cache()

            # 3. Build prediction text from logits (no real generation)
            pred_text = self.pos_label if logit_pos > logit_neg else self.neg_label

            return {
                "id": video_path,
                "answers": [pred_text],
                "logits": {self.neg_label: logit_neg, self.pos_label: logit_pos},
            }, None

        except Exception as e:
            return {"id": video_path, "answers": [f"Error: {str(e)}"]}, str(e)


# ====================== 主流程 ======================
def main_worker(rank, world_size, model_args, data_args, training_args):
    device = f"npu:{rank}"
    torch.npu.set_device(device)

    # Attention backend: default SDPA (works on NPU); override via INFER_ATTN_IMPL=flash_attention_2 / eager
    attn_impl = os.getenv("INFER_ATTN_IMPL", "sdpa")
    print(f"Rank {rank}: loading merged Qwen3.5 model from {model_args.model_id} (attn={attn_impl})")
    model = TargetVLModel.from_pretrained(
        model_args.model_id,
        torch_dtype="auto",
        device_map=None,
        attn_implementation=attn_impl,
    ).eval().to(device)

    processor = AutoProcessor.from_pretrained(model_args.model_id)

    # ---- Load classification head ----
    head_bin = Path(model_args.model_id) / "cls_head.bin"
    head_meta = Path(model_args.model_id) / "cls_head_meta.json"
    if not head_bin.exists() or not head_meta.exists():
        raise FileNotFoundError(
            f"cls_head.bin / cls_head_meta.json not found in {model_args.model_id}. "
            "Make sure your merge step copied them from the checkpoint dir."
        )
    with open(head_meta, encoding="utf-8") as f:
        meta = json.load(f)
    # Backward compat: older meta variants
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

    # Optional: read video params from training_args / env if you exposed them; default matches v2
    video_fps = float(os.getenv("INFER_VIDEO_FPS", "8.0"))
    video_max_pixels = int(os.getenv("INFER_VIDEO_MAX_PIXELS", "602112"))

    model_pool = ModelPool(
        model, processor, cls_head, meta["label_map"], device,
        video_fps=video_fps, video_max_pixels=video_max_pixels,
    )

    # ---- File paths & resume ----
    output_file = f"{data_args.output_json.rsplit('.', 1)[0]}_rank{rank}.json"
    completed_ids = set()
    results = []
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                results = json.load(f)
                completed_ids = {r["id"] for r in results}
        except Exception:
            pass

    # ---- Task distribution (rank 0 dispatches) ----
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
        task_lists = [[] for _ in range(world_size)]
        for size, sample, video_path in sample_sizes:
            load, w = heapq.heappop(worker_loads)
            task_lists[w].append((sample, video_path))
            heapq.heappush(worker_loads, (load + size, w))

        dist_obj = [task_lists]
    else:
        dist_obj = [None]

    dist.broadcast_object_list(dist_obj, src=0)
    my_tasks = dist_obj[0][rank]
    my_queue = [t for t in my_tasks if t[1] not in completed_ids]
    print(f"Rank {rank}: {len(my_queue)} tasks, parallel workers={PARALLEL_WORKERS}")

    counter = len(results)
    save_step = 10

    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        future_to_path = {}
        for messages, video_path in my_queue:
            if not os.path.exists(video_path):
                results.append({"id": video_path, "answers": ["Error: Video not found"]})
                continue
            future = executor.submit(model_pool.infer_single, messages, video_path)
            future_to_path[future] = video_path

        pbar = tqdm(total=len(future_to_path), desc=f"Rank {rank}", position=rank)
        for future in as_completed(future_to_path):
            video_path = future_to_path[future]
            try:
                res, err = future.result()
                results.append(res)
                if err:
                    print(f"Rank {rank} Error {video_path}: {err}")
            except Exception as e:
                print(f"Rank {rank} Critical Error: {e}")
                results.append({"id": video_path, "answers": [f"Critical Error: {str(e)}"]})

            counter += 1
            pbar.update(1)
            if counter % save_step == 0:
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
        pbar.close()

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Rank {rank}: local inference done.")


def merge_results(final_output_path, world_size):
    all_results_dict = {}
    if os.path.exists(final_output_path):
        try:
            with open(final_output_path, "r", encoding="utf-8") as f:
                for item in json.load(f):
                    all_results_dict[item["id"]] = item
        except Exception as e:
            print(f"Warning reading existing aggregate: {e}")

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
                print(f"Failed to merge {temp_file}: {e}")

    final_list = list(all_results_dict.values())
    with open(final_output_path, "w", encoding="utf-8") as f:
        json.dump(final_list, f, ensure_ascii=False, indent=2)
    print(f"Aggregate done: {len(final_list)} unique entries.")


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
