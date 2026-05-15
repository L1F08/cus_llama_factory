#!/usr/bin/bash
# 验证 processor() 在 8 进程并发下的真实耗时。
# 单进程跑 batch=16 是 5.68s；如果 8 进程并发也接近这个数，说明竞争不是瓶颈；
# 如果飙到 30s+，就石锤了"跨进程内存/CPU 竞争"是导致主线程 build_inputs 慢的根因。

cd /home/ma-user/work/lyf/

# 清旧日志
rm -f /tmp/bench_*.log

# 并发起 8 个 batch_test.py
echo "starting 8 processes in parallel..."
START=$(date +%s)
for i in 0 1 2 3 4 5 6 7; do
  python batch_test.py > /tmp/bench_$i.log 2>&1 &
done

# 等所有完成
wait
END=$(date +%s)
echo "all 8 processes finished in $((END - START))s wall time"

echo ""
echo "=== batched batch=16 timing per process ==="
for i in 0 1 2 3 4 5 6 7; do
  printf "  proc %d: " $i
  grep "batched" /tmp/bench_$i.log || echo "(no result — check /tmp/bench_$i.log for errors)"
done

echo ""
echo "=== single-sample x 16 serial timing per process (for reference) ==="
for i in 0 1 2 3 4 5 6 7; do
  printf "  proc %d: " $i
  grep "single-sample" /tmp/bench_$i.log || echo "(no result)"
done
