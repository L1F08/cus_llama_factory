# import os
# import json
# import threading
# import heapq
# import time
# from tqdm import tqdm
# import torch
# import torch_npu
# import torch.distributed as dist
# from concurrent.futures import ThreadPoolExecutor, as_completed

# # # [Qwen3 专用引入]
# # try:
# #     from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, HfArgumentParser
# # except ImportError:
# #     from transformers import AutoModelForImageTextToText as Qwen3VLForConditionalGeneration, AutoProcessor, HfArgumentParser


# # [修改后：Qwen3.5 通用引入]
# try:
#     from transformers import Qwen3_5VLForConditionalGeneration as TargetVLModel, AutoProcessor, HfArgumentParser
# except ImportError:
#     # 强烈建议使用 AutoModelForImageTextToText 作为兜底，它可以自动路由 Dense 或 MoE 模型架构
#     from transformers import AutoModelForImageTextToText as TargetVLModel, AutoProcessor, HfArgumentParser





# from infer_params import DataArguments, ModelArguments, TrainingArguments
# from qwen_vl_utils import process_vision_info

# # ====================== 配置参数 ======================
# # 并行 Worker 数：Qwen3 的图像处理比 Qwen2.5 更重，建议从 4-6 开始尝试，避免内存溢出
# PARALLEL_WORKERS = 4 

# torch.npu.config.allow_internal_format = False
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"

# # ====================== Patch 逻辑 (保持不变) ======================
# # def apply_pixel_limit_patch():
# #     """Qwen3 像素限制 Patch"""
# #     try:
# #         pv_globals = process_vision_info.__globals__
# #         if 'VIDEO_MAX_TOKEN_NUM' in pv_globals:
# #             print(f"🔍 找到变量，原值 VIDEO_MAX_TOKEN_NUM = {pv_globals['VIDEO_MAX_TOKEN_NUM']}")
# #             pv_globals['VIDEO_MAX_TOKEN_NUM'] = 12000 
# #             pv_globals['MODEL_SEQ_LEN'] = 1000000 
# #             print(f"✅ Patch 成功! 新值: VIDEO_MAX_TOKEN_NUM={pv_globals['VIDEO_MAX_TOKEN_NUM']}")
# #         else:
# #             print("❌ Patch 警告: 未找到 VIDEO_MAX_TOKEN_NUM")
# #     except Exception as e:
# #         print(f"❌ Patch 失败: {e}")

# # apply_pixel_limit_patch()


# # ====================== 模型池 (核心加速类) ======================
# # ====================== 模型池 (核心加速类) ======================
# class ModelPool:
#     """
#     管理模型推理的线程安全池。
#     将 CPU 密集型的 process_vision_info 和 NPU 密集型的 model.generate 解耦。
#     """
#     def __init__(self, model, processor, device):
#         self.model = model
#         self.processor = processor
#         self.device = device
#         self.lock = threading.Lock()
        
#         # 【新增】获取 '0' 和 '1' 在词表中的 Token ID
#         # 使用 add_special_tokens=False 确保只获取纯数字的 ID
#         self.token_id_0 = self.processor.tokenizer.encode("0", add_special_tokens=False)[-1]
#         self.token_id_1 = self.processor.tokenizer.encode("1", add_special_tokens=False)[-1]
        
#         # Qwen3 生成参数
#         self.gen_kwargs = {
#             "max_new_tokens": 4096,  
#             "do_sample": False,
#             "temperature": 0.1,
#             "top_p": 0.95,
#             "repetition_penalty": 1.00,
#             # 【新增】让 generate 函数返回 logits 等详细信息
#             "return_dict_in_generate": True, 
#             "output_scores": True
#         }

#     def infer_single(self, messages, video_path):
#         """单个样本的处理流水线"""
#         try:
#             # 1. CPU 阶段：预处理 (无锁并行)
#             for msg in messages:
#                 if msg.get("role") == "user":
#                     for content_item in msg.get("content", []):
#                         if content_item.get("type") == "video":
#                             content_item["fps"] = 3.0
#                             content_item["max_pixels"] = 602112

#             # 应用聊天模板
#             text = self.processor.apply_chat_template(
#                 messages, 
#                 tokenize=False, 
#                 add_generation_prompt=True,
#                 enable_thinking=False  # 关闭自动补全 <think> 标签，确保第一个生成的 Token 就是答案
#             )
            
#             # 视觉处理
#             image_inputs, video_inputs, video_kwargs = process_vision_info(
#                 messages,
#                 return_video_kwargs=True,
#                 image_patch_size=16,        
#                 return_video_metadata=True, 
#             )

#             video_metadatas = None
#             if video_inputs is not None and len(video_inputs) > 0:
#                 if isinstance(video_inputs, (tuple, list)): 
#                     video_inputs, video_metadatas = zip(*video_inputs)
#                     video_inputs = list(video_inputs)
#                     video_metadatas = list(video_metadatas)

#             # 2. NPU 阶段：推理 (加锁串行)
#             with self.lock:
#                 inputs = self.processor(
#                     text=[text],
#                     images=image_inputs,
#                     videos=video_inputs,
#                     video_metadata=video_metadatas,
#                     padding=True,
#                     return_tensors="pt",
#                     **video_kwargs
#                 ).to(self.device)

#                 # 生成
#                 with torch.no_grad():
#                     # 【修改】获取完整的输出字典对象
#                     outputs = self.model.generate(**inputs, **self.gen_kwargs)

#                 # 3. 后处理：解码与 Logits 提取 (持有锁的时间极短)
#                 generated_ids = outputs.sequences
                
#                 # 【新增】提取第一个生成的 token 的 Logits
#                 # outputs.scores 是一个 tuple，长度为生成的 token 数量
#                 # outputs.scores[0] 是第一个生成 token 的预测分布，shape 为 (batch_size, vocab_size)
#                 first_step_logits = outputs.scores[0][0]  # 因为 batch_size=1，所以取 [0]
                
#                 # 获取 0 和 1 对应的对数值 (转为 float 以便 JSON 序列化)
#                 logit_0 = float(first_step_logits[self.token_id_0].item())
#                 logit_1 = float(first_step_logits[self.token_id_1].item())

#                 # 剔除 prompt 部分的 input ids
#                 generated_ids_trimmed = [
#                     out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
#                 ]
                
#                 # 显存清理
#                 del inputs, outputs, generated_ids, first_step_logits
#                 torch.npu.empty_cache()

#             # 4. CPU 阶段：Decode (无锁并行)
#             out_text = self.processor.batch_decode(
#                 generated_ids_trimmed, 
#                 skip_special_tokens=True, 
#                 clean_up_tokenization_spaces=False
#             )
            
#             # 【修改】将 logits 加入最终的返回字典中
#             return {
#                 "id": video_path, 
#                 "answers": out_text, 
#                 "logits": {"0": logit_0, "1": logit_1}
#             }, None

#         except Exception as e:
#             return {"id": video_path, "answers": [f"Error: {str(e)}"]}, str(e)

# # ====================== 主工作流 ======================
# def main_worker(rank, world_size, model_args, data_args, training_args):
#     device = f'npu:{rank}'
#     torch.npu.set_device(device)
    
#     # 1. 加载模型
#     # [修改后]
#     print(f"Rank {rank}: 加载 Qwen3.5 模型...")
#     model = TargetVLModel.from_pretrained(
#         model_args.model_id,
#         torch_dtype="auto",
#         device_map=None,
#         attn_implementation="flash_attention_2"
#     ).eval().to(device)


    
#     processor = AutoProcessor.from_pretrained(model_args.model_id)
#     model_pool = ModelPool(model, processor, device)

#     # 2. 准备文件路径
#     output_file = f"{data_args.output_json.rsplit('.', 1)[0]}_rank{rank}.json"
    
#     # 3. 读取已完成结果
#     completed_ids = set()
#     if os.path.exists(output_file):
#         try:
#             with open(output_file, 'r', encoding='utf-8') as f:
#                 existing = json.load(f)
#                 completed_ids = {r['id'] for r in existing}
#         except: pass
        
#     results = []
#     if os.path.exists(output_file):
#         with open(output_file, 'r') as f: results = json.load(f)

#     # 4. 数据分配 (仅 Rank 0 读取并分发)
#     tasks = []
#     if rank == 0:
#         with open(data_args.data_path, 'r', encoding='utf-8') as f:
#             test_samples = json.load(f)

#         # 预处理数据结构
#         valid_samples = []
#         for sample in test_samples:
#             try:
#                 # 适配 [[{...}]] 结构
#                 first_msg = sample[0] if isinstance(sample, list) else sample
#                 content = first_msg.get("content", [])
#                 video_path = next((item["video"] for item in content if item["type"] == "video"), None)
                
#                 if video_path and video_path not in completed_ids: # 注意这里只在Rank0过滤最终的大表，具体rank内过滤在下面
#                      valid_samples.append((sample, video_path))
#             except: continue

#         # 按大小排序
#         sample_sizes = []
#         for sample, video_path in valid_samples:
#             size = os.path.getsize(video_path) if os.path.exists(video_path) else 0
#             sample_sizes.append((size, sample, video_path))
#         sample_sizes.sort(key=lambda x: x[0], reverse=True)

#         # 堆分配
#         worker_loads = [(0, w) for w in range(world_size)]
#         heapq.heapify(worker_loads)
#         task_lists = [[] for _ in range(world_size)]
        
#         for size, sample, video_path in sample_sizes:
#             load, w = heapq.heappop(worker_loads)
#             task_lists[w].append((sample, video_path))
#             heapq.heappush(worker_loads, (load + size, w))
            
#         dist_obj = [task_lists]
#     else:
#         dist_obj = [None]

#     # 广播任务
#     dist.broadcast_object_list(dist_obj, src=0)
#     my_tasks = dist_obj[0][rank]
    
#     # 过滤掉本Rank已经跑过的 (Double check)
#     my_queue = [t for t in my_tasks if t[1] not in completed_ids]

#     print(f"Rank {rank}: 分配到 {len(my_queue)} 个任务，并行数 {PARALLEL_WORKERS}")

#     # 5. 并行推理执行
#     counter = len(results)
#     save_step = 10
    
#     with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
#         future_to_path = {}
#         for messages, video_path in my_queue:
#             if not os.path.exists(video_path):
#                 results.append({"id": video_path, "answers": ["Error: Video not found"]})
#                 continue
#             future = executor.submit(model_pool.infer_single, messages, video_path)
#             future_to_path[future] = video_path

#         # 进度条
#         pbar = tqdm(total=len(future_to_path), desc=f"Rank {rank}", position=rank)
        
#         for future in as_completed(future_to_path):
#             video_path = future_to_path[future]
#             try:
#                 res, err = future.result()
#                 results.append(res)
#                 if err:
#                     print(f"Rank {rank} Error {video_path}: {err}")
#             except Exception as e:
#                 print(f"Rank {rank} Critical Error: {e}")
            
#             counter += 1
#             pbar.update(1)
            
#             if counter % save_step == 0:
#                 with open(output_file, 'w', encoding='utf-8') as f:
#                     json.dump(results, f, ensure_ascii=False, indent=2)
        
#         pbar.close()

#     # 最终保存
#     with open(output_file, 'w', encoding='utf-8') as f:
#         json.dump(results, f, ensure_ascii=False, indent=2)
#     print(f"Rank {rank}: 全部完成。")


# def main():
#     parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
#     model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
#     local_rank = int(os.getenv("LOCAL_RANK", 0))
#     world_size = int(os.getenv("WORLD_SIZE", 1))
    
#     os.environ["HCCL_CONNECT_TIMEOUT"] = "7200"
    
#     if not dist.is_initialized():
#         dist.init_process_group(backend="hccl", world_size=world_size, rank=local_rank)
    
#     main_worker(local_rank, world_size, model_args, data_args, training_args)
    
#     if dist.is_initialized():
#         dist.destroy_process_group()
        
#     if local_rank == 0:
#         # 简单的合并逻辑
#         import time
#         time.sleep(2)
#         merge_results(data_args.output_json, world_size)

# def merge_results(final_output_path, world_size):
#     all_results = []
#     base_name = final_output_path.rsplit('.', 1)[0]
#     for r in range(world_size):
#         temp_file = f"{base_name}_rank{r}.json"
#         if os.path.exists(temp_file):
#             with open(temp_file, 'r') as f:
#                 all_results.extend(json.load(f))
#             os.remove(temp_file) # 可选：删除临时文件
    
#     with open(final_output_path, 'w', encoding='utf-8') as f:
#         json.dump(all_results, f, ensure_ascii=False, indent=2)
#     print(f"合并完成，共 {len(all_results)} 条数据")

# if __name__ == "__main__":
#     main()



import os
import json
import threading
import heapq
import time
from tqdm import tqdm
import torch
import torch_npu
import torch.distributed as dist
from concurrent.futures import ThreadPoolExecutor, as_completed

# [修改后：Qwen3.5 通用引入]
try:
    from transformers import Qwen3_5VLForConditionalGeneration as TargetVLModel, AutoProcessor, HfArgumentParser
except ImportError:
    # 强烈建议使用 AutoModelForImageTextToText 作为兜底，它可以自动路由 Dense 或 MoE 模型架构
    from transformers import AutoModelForImageTextToText as TargetVLModel, AutoProcessor, HfArgumentParser

from infer_params import DataArguments, ModelArguments, TrainingArguments
from qwen_vl_utils import process_vision_info

# ====================== 配置参数 ======================
# 并行 Worker 数：Qwen3 的图像处理比 Qwen2.5 更重，建议从 4-6 开始尝试，避免内存溢出
PARALLEL_WORKERS = 6

torch.npu.config.allow_internal_format = False
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"

# ====================== 模型池 (核心加速类) ======================
class ModelPool:
    """
    管理模型推理的线程安全池。
    将 CPU 密集型的 process_vision_info 和 NPU 密集型的 model.generate 解耦。
    """
    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device
        self.lock = threading.Lock()
        
        # 获取 '0' 和 '1' 在词表中的 Token ID
        # 使用 add_special_tokens=False 确保只获取纯数字的 ID
        # self.token_id_0 = self.processor.tokenizer.encode("0", add_special_tokens=False)[-1]
        # self.token_id_1 = self.processor.tokenizer.encode("1", add_special_tokens=False)[-1]
        self.token_id_safe = self.processor.tokenizer.encode("安全", add_special_tokens=False)[0]
        self.token_id_risk = self.processor.tokenizer.encode("高风险", add_special_tokens=False)[0]
        # Qwen3 生成参数
        # IMPORTANT: max_new_tokens=1 instead of 4096.
        # Reason: 我们只用 outputs.scores[0] (第一步 logits)，根本不需要后续生成。
        # 旧版的 4096 在「strip <think></think> 块」之后会触发 Qwen3.5 的 reasoning
        # 习惯（pre-training 学到的，fine-tune 数据无法完全压制），模型会生成几百到
        # 上千个 token 的内部"思考"才输出答案，导致单样本耗时 10x+。
        # 既然只读第一步 logits，max_new_tokens=1 就够了 —— 不会改变 logit 值。
        self.gen_kwargs = {
            "max_new_tokens": 1,
            "do_sample": False,
            "temperature": 0.1,
            "top_p": 0.95,
            "repetition_penalty": 1.00,
            "return_dict_in_generate": True,
            "output_scores": True,
            "pad_token_id": self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id
        }

    def infer_single(self, messages, video_path):
        """单个样本的处理流水线"""
        try:
            # 1. CPU 阶段：预处理 (无锁并行)
            for msg in messages:
                if msg.get("role") == "user":
                    for content_item in msg.get("content", []):
                        if content_item.get("type") == "video":
                            content_item["fps"] = 8.0
                            content_item["max_pixels"] = 602112

            # 应用聊天模板
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False  # 让模板走 nothink 分支
            )
            # 关键修复：Qwen3.5-VL 的 chat_template.jinja 即使 enable_thinking=False
            # 也会在 assistant 标签后插入空的 "<think>\n\n</think>\n\n" 块。
            # 但训练用的 qwen3_5_nothink 模板根本不会加这个块——这导致 generate()
            # 的起始位置 OOD（训练时是 assistant\n，推理时是 </think>\n\n），
            # logits 严重偏离训练分布，best threshold 被压到 ~0.1。
            # Strip 后让推理 prompt 与训练完全对齐。
            text = text.replace("<think>\n\n</think>\n\n", "")
            
            # 视觉处理
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

            # 2. NPU 阶段：推理 (加锁串行)
            with self.lock:
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    video_metadata=video_metadatas,
                    padding=True,
                    return_tensors="pt",
                    **video_kwargs
                ).to(self.device)

                # 生成
                with torch.no_grad():
                    # 获取完整的输出字典对象
                    outputs = self.model.generate(**inputs, **self.gen_kwargs)

                # 3. 后处理：解码与 Logits 提取 (持有锁的时间极短)
                generated_ids = outputs.sequences
                
                # 提取第一个生成的 token 的 Logits
                first_step_logits = outputs.scores[0][0]  # 因为 batch_size=1，所以取 [0]
                
                # # 获取 0 和 1 对应的对数值 (转为 float 以便 JSON 序列化)
                # logit_0 = float(first_step_logits[self.token_id_0].item())
                # logit_1 = float(first_step_logits[self.token_id_1].item())
                # 获取“安全”和“高风险”对应的对数值
                logit_safe = float(first_step_logits[self.token_id_safe].item())
                logit_risk = float(first_step_logits[self.token_id_risk].item())
                # 剔除 prompt 部分的 input ids
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                
                # 显存清理
                del inputs, outputs, generated_ids, first_step_logits
                torch.npu.empty_cache()

            # 4. CPU 阶段：Decode (无锁并行)
            out_text = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
            
            # # 将 logits 加入最终的返回字典中
            # return {
            #     "id": video_path, 
            #     "answers": out_text, 
            #     "logits": {"0": logit_0, "1": logit_1}
            # }, None
            # 将 logits 加入最终的返回字典中
            return {
                "id": video_path, 
                "answers": out_text, 
                "logits": {"安全": logit_safe, "高风险": logit_risk}
            }, None
        except Exception as e:
            return {"id": video_path, "answers": [f"Error: {str(e)}"]}, str(e)

# ====================== 主工作流 ======================
def main_worker(rank, world_size, model_args, data_args, training_args):
    device = f'npu:{rank}'
    torch.npu.set_device(device)
    
    # 1. 加载模型
    print(f"Rank {rank}: 加载 Qwen3.5 模型...")
    model = TargetVLModel.from_pretrained(
        model_args.model_id,
        torch_dtype="auto",
        device_map=None,
        attn_implementation="flash_attention_2"
    ).eval().to(device)

    processor = AutoProcessor.from_pretrained(model_args.model_id)
    model_pool = ModelPool(model, processor, device)

    # 2. 准备文件路径
    output_file = f"{data_args.output_json.rsplit('.', 1)[0]}_rank{rank}.json"
    
    # 3. 读取本地已完成结果 (针对当前 Rank 的中间文件)
    completed_ids = set()
    results = []
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
                completed_ids = {r['id'] for r in results}
        except: pass

    # 4. 数据分配 (仅 Rank 0 读取并分发)
    if rank == 0:
        # [优化]: Rank 0 读取全局最终文件，确保分配时不会重复分配以前跑完的全局数据
        global_completed_ids = set()
        if os.path.exists(data_args.output_json):
            try:
                with open(data_args.output_json, 'r', encoding='utf-8') as f:
                    global_existing = json.load(f)
                    global_completed_ids = {r['id'] for r in global_existing}
            except: pass

        with open(data_args.data_path, 'r', encoding='utf-8') as f:
            test_samples = json.load(f)

        # 预处理数据结构
        valid_samples = []
        for sample in test_samples:
            try:
                # 适配 [[{...}]] 结构
                first_msg = sample[0] if isinstance(sample, list) else sample
                content = first_msg.get("content", [])
                video_path = next((item["video"] for item in content if item["type"] == "video"), None)
                
                # [修复]: 结合全局已完成表进行过滤，避免重复分配
                if video_path and video_path not in global_completed_ids and video_path not in completed_ids: 
                     valid_samples.append((sample, video_path))
            except: continue

        # 按大小排序
        sample_sizes = []
        for sample, video_path in valid_samples:
            size = os.path.getsize(video_path) if os.path.exists(video_path) else 0
            sample_sizes.append((size, sample, video_path))
        sample_sizes.sort(key=lambda x: x[0], reverse=True)

        # 堆分配
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
    
    # 过滤掉本Rank已经跑过的 (Double check)
    my_queue = [t for t in my_tasks if t[1] not in completed_ids]

    print(f"Rank {rank}: 分配到 {len(my_queue)} 个任务，并行数 {PARALLEL_WORKERS}")

    # 5. 并行推理执行
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

        # 进度条
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
                # [修复]: 将致命错误的记录追加进 results，防止总数丢失截断
                results.append({"id": video_path, "answers": [f"Critical Error: {str(e)}"]})
            
            counter += 1
            pbar.update(1)
            
            if counter % save_step == 0:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
        
        pbar.close()

    # 最终保存
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Rank {rank}: 本地推理全部完成。")


def merge_results(final_output_path, world_size):
    """
    [修复]: 优化合并逻辑，采用字典根据 ID 去重，并且如果最终大文件已存在，先将其读入，防止覆盖历史数据。
    """
    all_results_dict = {}
    
    # 1. 读入已有的全局最终结果（如果是断点续传的话，保留原有的）
    if os.path.exists(final_output_path):
        try:
            with open(final_output_path, 'r', encoding='utf-8') as f:
                for item in json.load(f):
                    all_results_dict[item['id']] = item
        except Exception as e:
            print(f"读取历史汇总文件警告: {e}")

    # 2. 合并各个 Rank 的临时文件
    base_name = final_output_path.rsplit('.', 1)[0]
    for r in range(world_size):
        temp_file = f"{base_name}_rank{r}.json"
        if os.path.exists(temp_file):
            try:
                with open(temp_file, 'r', encoding='utf-8') as f:
                    for item in json.load(f):
                        all_results_dict[item['id']] = item  # 同 ID 会被覆盖，保证唯一性
                os.remove(temp_file) # 读取成功后删除临时文件
            except Exception as e:
                print(f"合并 {temp_file} 失败: {e}")
    
    # 3. 输出最终文件
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
    
    # 执行推理工作流
    main_worker(local_rank, world_size, model_args, data_args, training_args)
    
    if dist.is_initialized():
        # [修复]: 新增 barrier()，强制所有进程等待，确保所有人写完文件后再销毁进程组合并
        dist.barrier()
        dist.destroy_process_group()
        
    if local_rank == 0:
        # [修复]: 删除 time.sleep(2)，因为 barrier 已经彻底保证了同步安全
        merge_results(data_args.output_json, world_size)

if __name__ == "__main__":
    main()


    