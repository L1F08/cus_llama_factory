"""
扫多组 (BATCH_SIZE, PREPROCESS_WORKERS) 配置，找单卡最高效推理设置。

要点：
  - 模型只加载一次，所有配置复用 → 扫一大堆配置也不用反复等加载
  - 复用生产代码路径（import infer_faster_qwen35_logits_nat_batch 的
    preprocess_sample / _collate_qwen_vl / BatchedLogitsInferencer / 三段流水线）
    → 测的就是真实性能
  - **排除 warmup**：跳过前 WARMUP_SAMPLES 个样本，只对随后 MEASURE_SAMPLES 个
    计时 → 直接得到稳态 s/sample，不被前几个慢 batch 拖累（这正是你看诊断行
    前 3 行会被误导的原因）
  - 单卡(npu:0)单进程：测的是 per-rank 稳态吞吐。各 rank 独立，相对排序有代表性；
    绝对值会比 8 卡并发略乐观（跨卡内存带宽争抢），但用于"选哪个配置"足够。

用法（可跑很久，配置/样本数随便加）：
  cd /home/ma-user/work/lyf/
  python bench_infer_config.py
"""

import os
import sys
import json
import time
import threading
import queue as _queue
from collections import deque
from concurrent.futures import ThreadPoolExecutor

# ====== 改成你的实际路径 ======
SCRIPTS_DIR = "/home/ma-user/work/lyf"
MODEL_PATH = "/home/ma-user/work/lyf/outmodel/crash_1cam_2cls_train_3s_39k_0508-800_nothink-merge"
DATA_PATH = "/home/ma-user/work/lyf/data/0506_crash_1cam_2cls_test_39k_3s/test_0506_crash_1cam_2cls_test_39k_3s_front_with_ego_info_5256_3s_clipped_cleaned_dedup_4k.json"

# 要扫的配置网格： (BATCH_SIZE, PREPROCESS_WORKERS)
# 第二轮探索：上一轮发现瓶颈是 CPU 预处理(视频解码)，且 BS=16/PW=16 仍最快、未到平台。
# 这一轮做单变量对照，找平台 + 看到底是 batch 还是 worker 绑定：
CONFIGS = [
    # 固定 PW=16，加大 BATCH —— 看 batch 还能不能继续提速 / 何时 OOM
    (12, 16), (16, 16), (24, 16), (32, 16),
    # 固定 BATCH=16，加大 PW —— 看 worker 是否仍是绑定资源
    (16, 12), (16, 20), (16, 24),
    # 高端组合
    (24, 24), (32, 24),
]
WARMUP_SAMPLES = 60     # 跳过的预热样本数（不计时）
MEASURE_SAMPLES = 150   # 计时样本数（这一轮调小，单配置更快出结果；要更稳可加大）
LIVE_EVERY = 30         # 每处理这么多“计时样本”打一次瞬时速率
# 提示：视频解码是主瓶颈。若环境装了 decord，先在 shell 里
#   export FORCE_QWENVL_VIDEO_READER=decord
# 再跑本脚本，大概率比调 batch/worker 收益大得多（可对比加/不加的汇总表）。
# =============================

sys.path.insert(0, SCRIPTS_DIR)
import importlib
mod = importlib.import_module("infer_faster_qwen35_logits_nat_batch")

import torch
import torch_npu  # noqa

device = "npu:0"
torch.npu.set_device(device)

print(f"加载模型一次：{MODEL_PATH}")
model = mod.TargetVLModel.from_pretrained(
    MODEL_PATH, torch_dtype="auto", device_map=None,
    attn_implementation="flash_attention_2",
).eval().to(device)
processor = mod.AutoProcessor.from_pretrained(MODEL_PATH)
inferencer = mod.BatchedLogitsInferencer(model, processor, device)
print(f"model dtype = {next(model.parameters()).dtype}")

# 取足够的样本，按视频文件大小降序排（模拟 rank 内顺序）
need = WARMUP_SAMPLES + MEASURE_SAMPLES + 16
raw = json.load(open(DATA_PATH))
pool = []
for s in raw:
    try:
        first = s[0] if isinstance(s, list) else s
        vp = next((c["video"] for c in first.get("content", []) if c["type"] == "video"), None)
        if vp and os.path.exists(vp):
            pool.append((os.path.getsize(vp), s, vp))
    except Exception:
        continue
pool.sort(key=lambda x: x[0], reverse=True)
pool = pool[:need]
runnable_all = [(s, v) for _, s, v in pool]
print(f"用 {len(runnable_all)} 个样本（warmup {WARMUP_SAMPLES} + measure {MEASURE_SAMPLES}）")
if len(runnable_all) < WARMUP_SAMPLES + MEASURE_SAMPLES:
    print("⚠️ 样本不够，请减小 WARMUP/MEASURE 或换更大的数据集")
    sys.exit(1)


def run_one_config(BS, PW):
    """跑一遍 FIFO 三段流水线；返回 (steady_s_per_sample, status)。
    只对第 WARMUP..WARMUP+MEASURE 个完成的样本计时。"""
    npu_inbox = _queue.Queue(maxsize=2)
    npu_outbox = _queue.Queue()
    stop = object()

    def npu_worker():
        while True:
            it = npu_inbox.get()
            if it is stop:
                return
            batch, cpu_inputs = it
            try:
                br = inferencer.run_inference(batch, cpu_inputs)
            except BaseException as e:
                br = [{"id": x["video_path"], "answers": [f"ERR:{type(e).__name__}"]} for x in batch]
            npu_outbox.put(br)

    th = threading.Thread(target=npu_worker, daemon=True)
    th.start()

    done = 0                 # 已取回结果的样本数
    t_warm_end = None        # 第 WARMUP 个样本完成的时刻
    t_meas_end = None
    last_live = 0
    pending = []
    submitted = consumed = 0

    def drain(block_until=None):
        nonlocal consumed, done, t_warm_end, t_meas_end, last_live
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
            consumed += 1
            done += len(br)
            if t_warm_end is None and done >= WARMUP_SAMPLES:
                t_warm_end = time.perf_counter()
            if t_warm_end is not None and t_meas_end is None and done >= WARMUP_SAMPLES + MEASURE_SAMPLES:
                t_meas_end = time.perf_counter()
            if t_warm_end is not None and t_meas_end is None:
                m = done - WARMUP_SAMPLES
                if m - last_live >= LIVE_EVERY:
                    inst = m / (time.perf_counter() - t_warm_end)
                    print(f"    [BS={BS} PW={PW}] measured {m}/{MEASURE_SAMPLES}  "
                          f"inst={inst:.3f} samp/s ({1/inst:.2f} s/samp)", flush=True)
                    last_live = m

    try:
        with ThreadPoolExecutor(max_workers=PW) as ex:
            task_iter = iter(runnable_all)
            window = deque()
            max_inflight = max(PW * 2, BS + PW)

            def submit_next():
                try:
                    m, v = next(task_iter)
                except StopIteration:
                    return False
                window.append(ex.submit(mod.preprocess_sample, processor, m, v))
                return True

            for _ in range(max_inflight):
                if not submit_next():
                    break

            while window and t_meas_end is None:
                fut = window.popleft()
                submit_next()
                item = fut.result()
                if item.get("error"):
                    continue
                pending.append(item)
                if len(pending) >= BS:
                    cpu_inputs = inferencer.build_inputs_cpu(pending)
                    npu_inbox.put((pending, cpu_inputs))
                    submitted += 1
                    pending = []
                    drain(block_until=None)

            # 收尾把已提交的取回（保证 t_meas_end 被设上）
            drain(block_until=submitted)
    except BaseException as e:
        npu_inbox.put(stop)
        return None, f"{type(e).__name__}: {str(e)[:80]}"

    npu_inbox.put(stop)
    th.join(timeout=30)
    torch.npu.empty_cache()

    if t_warm_end is None or t_meas_end is None:
        return None, "样本不足以测满 MEASURE"
    sps = MEASURE_SAMPLES / (t_meas_end - t_warm_end)
    return (1.0 / sps), "OK"


rows = []
for BS, PW in CONFIGS:
    print(f"\n==== 测试 BATCH_SIZE={BS}  PREPROCESS_WORKERS={PW} ====", flush=True)
    t0 = time.perf_counter()
    spp, status = run_one_config(BS, PW)
    wall = time.perf_counter() - t0
    if spp is None:
        print(f"  -> {status}  (本次耗时 {wall:.0f}s)")
        rows.append((BS, PW, None, None, status))
    else:
        print(f"  -> 稳态 {spp:.3f} s/sample  ({1/spp:.3f} samples/s)  (本次耗时 {wall:.0f}s)")
        rows.append((BS, PW, spp, 1.0 / spp, "OK"))

print("\n================ 汇总（单卡稳态，已排除 warmup）================")
print(f"{'BATCH':>6} {'WORKERS':>8} {'s/sample':>10} {'samples/s':>11}  status")
best = None
for BS, PW, spp, sps, st in rows:
    if spp is None:
        print(f"{BS:>6} {PW:>8} {'-':>10} {'-':>11}  {st}")
    else:
        print(f"{BS:>6} {PW:>8} {spp:>10.3f} {sps:>11.3f}  {st}")
        if best is None or sps > best[3]:
            best = (BS, PW, spp, sps)

if best:
    BS, PW, spp, sps = best
    print(f"\n最快配置：BATCH_SIZE={BS} PREPROCESS_WORKERS={PW} "
          f"({spp:.3f} s/sample)")
    # 找“在最快 3% 以内、BATCH 最小”的配置（省显存/warmup 快/批量精度更近单样本）
    cand = [r for r in rows if r[2] is not None and r[3] >= sps * 0.97]
    cand.sort(key=lambda r: (r[0], r[1]))
    rec = cand[0]
    print(f"推荐配置（性能在最快 3% 内、BATCH 最小）："
          f"BATCH_SIZE={rec[0]} PREPROCESS_WORKERS={rec[1]} "
          f"({rec[2]:.3f} s/sample)")
    print("说明：NPU-bound 时各配置通常接近平台；选小 BATCH 更省显存、"
          "warmup 更快、批量结果更接近单样本。")
