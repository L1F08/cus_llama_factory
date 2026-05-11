import json
import os
import math
import random
import shutil
from pathlib import Path

# 导入 sklearn 用于计算 ROC-AUC 和 全局最佳阈值
try:
    from sklearn.metrics import roc_auc_score, precision_recall_curve
    HAS_SKLEARN = True
except ImportError:
    print("⚠️ 未找到 scikit-learn，请先执行: pip install scikit-learn")
    HAS_SKLEARN = False

def calculate_metrics_with_logits(pred_json_path, gt_jsonl_path, fn_output_dir=None, fp_output_dir=None, sample_num=10, target_threshold=0.5):
    """
    基于 Logits 评估不同阈值下的模型表现，并寻找全局真正最佳阈值
    """
    # 打印当前评测的预测文件
    print("="*85)
    print(f"📂 当前评测的预测文件:\n{pred_json_path}")
    print("="*85)
    print("加载并解析数据中...")
    
    # 1. 加载真实标签 (Ground Truth)
    gt_dict = {}
    with open(gt_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): 
                continue
            data = json.loads(line.strip())
            
            raw_gt = str(data['label']).strip()
            if raw_gt == "高风险":
                gt_label = 1
            elif raw_gt == "安全":
                gt_label = 0
            else:
                gt_label = int(raw_gt)
                
            gt_dict[str(data['id'])] = gt_label
            
    # 2. 加载预测结果
    with open(pred_json_path, 'r', encoding='utf-8') as f:
        preds = json.load(f)
        
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    metrics_per_thresh = {t: {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0} for t in thresholds}
    
    missing_ids = []
    parse_errors = [] 
    fn_video_paths_target = [] # 漏报 (真1预0)
    fp_video_paths_target = [] # 误报 (真0预1)
    
    valid_count = 0
    
    # 用于计算 ROC-AUC 和 寻找真正最佳阈值的列表
    y_true_all = []
    y_scores_all = []

    # 3. 遍历核对
    for p in preds:
        original_video_path = p['id']  
        video_id = Path(original_video_path).stem  
        
        if video_id not in gt_dict:
            missing_ids.append(video_id)
            continue
            
        true_label = gt_dict[video_id]
        logits = p.get('logits', None)
        if not logits or '安全' not in logits or '高风险' not in logits:
            parse_errors.append((video_id, "Missing or invalid logits"))
            continue
            
        try:
            logit_safe = float(logits['安全'])
            logit_risk = float(logits['高风险'])
        except ValueError:
            parse_errors.append((video_id, "Logits are not numbers"))
            continue

        # Softmax 计算 P(y=1)
        max_logit = max(logit_safe, logit_risk)
        exp_safe = math.exp(logit_safe - max_logit)
        exp_risk = math.exp(logit_risk - max_logit)
        prob_risk = exp_risk / (exp_safe + exp_risk)
        
        valid_count += 1
        
        # 记录真实标签和预测概率
        y_true_all.append(true_label)
        y_scores_all.append(prob_risk)

        # 4. 统计固定分布阈值下的表现 (用于打印表格)
        for t in thresholds:
            pred_label = 1 if prob_risk >= t else 0
            
            if true_label == 1 and pred_label == 1:
                metrics_per_thresh[t]['TP'] += 1
            elif true_label == 0 and pred_label == 1:
                metrics_per_thresh[t]['FP'] += 1
                # 记录误报视频 (FP)
                if math.isclose(t, target_threshold):
                    fp_video_paths_target.append(original_video_path)
            elif true_label == 0 and pred_label == 0:
                metrics_per_thresh[t]['TN'] += 1
            elif true_label == 1 and pred_label == 0:
                metrics_per_thresh[t]['FN'] += 1
                # 记录漏报视频 (FN)
                if math.isclose(t, target_threshold):
                    fn_video_paths_target.append(original_video_path)

    if valid_count == 0:
        print("❌ 没有找到有效数据。")
        return

    # 5. 计算 ROC-AUC
    auc_score = None
    if HAS_SKLEARN and len(set(y_true_all)) > 1:
        auc_score = roc_auc_score(y_true_all, y_scores_all)

    # 6. 精确寻找全局最佳阈值 (基于所有可能出现的概率值)
    best_exact_f1 = -1
    best_exact_thresh = 0.5
    
    if HAS_SKLEARN and len(set(y_true_all)) > 1:
        # precision_recall_curve 会穷举所有能改变 TP/FP 状态的阈值
        precisions, recalls, pr_thresholds = precision_recall_curve(y_true_all, y_scores_all)
        for i in range(len(pr_thresholds)):
            p = precisions[i]
            r = recalls[i]
            if p + r > 0:
                f1 = 2 * p * r / (p + r)
                if f1 > best_exact_f1:
                    best_exact_f1 = f1
                    best_exact_thresh = pr_thresholds[i]

    # 根据找到的真正最佳阈值，单独算一次 TP/FP/TN/FN 以便打印
    best_TP, best_FP, best_TN, best_FN = 0, 0, 0, 0
    for true_y, score in zip(y_true_all, y_scores_all):
        pred_y = 1 if score >= best_exact_thresh else 0
        if true_y == 1 and pred_y == 1: best_TP += 1
        elif true_y == 0 and pred_y == 1: best_FP += 1
        elif true_y == 0 and pred_y == 0: best_TN += 1
        elif true_y == 1 and pred_y == 0: best_FN += 1

    best_acc = (best_TP + best_TN) / valid_count
    best_prec = best_TP / (best_TP + best_FP) if (best_TP + best_FP) > 0 else 0.0
    best_rec = best_TP / (best_TP + best_FN) if (best_TP + best_FN) > 0 else 0.0

    # 7. 打印报表
    print("\n" + "="*85)
    print(f"📊 模型评估报告 | 基准抽样阈值: {target_threshold} | 总有效样本: {valid_count}")
    if auc_score is not None:
        print(f"📈 整体 ROC-AUC Score: {auc_score:.4f}")
    print("="*85)
    print("--- 趋势概览 (固定步长) ---")
    print(f"{'Threshold':<10} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | TP   FP   TN   FN")
    print("-" * 85)
    
    for t in thresholds:
        m = metrics_per_thresh[t]
        TP, FP, TN, FN = m['TP'], m['FP'], m['TN'], m['FN']
        accuracy = (TP + TN) / valid_count
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
        marker = "⭐" if math.isclose(t, target_threshold) else "  "
        print(f"{marker} {t:<7.2f} | {accuracy:<10.2%} | {precision:<10.2%} | {recall:<10.2%} | {f1_score:<10.4f} | {TP:<4} {FP:<4} {TN:<4} {FN:<4}")
    
    print("-" * 85)
    print(f"🏆 全局真正最佳阈值 (Global Best F1-Score): {best_exact_thresh:.4f}")
    print(f"   ▶ Accuracy:  {best_acc:.2%}")
    print(f"   ▶ Precision: {best_prec:.2%}")
    print(f"   ▶ Recall:    {best_rec:.2%}")
    print(f"   ▶ F1-Score:  {best_exact_f1:.4f}")
    print(f"   ▶ 分布详情:  TP:{best_TP:<4}  FP:{best_FP:<4}  TN:{best_TN:<4}  FN:{best_FN:<4}")
    print("="*85)

    # 8. 辅助函数：复制视频
    def copy_sampled_videos(video_list, output_dir, label_type):
        if not output_dir or not video_list:
            print(f"ℹ️ 无需抽取或没有发现 {label_type} 样本。")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        actual_num = min(sample_num, len(video_list))
        sampled_list = random.sample(video_list, actual_num)
        
        print(f"\n📂 正在基于阈值 {target_threshold} 抽取 {actual_num} 个 {label_type} 视频到: {output_dir}")
        success = 0
        for vid_path in sampled_list:
            if os.path.exists(vid_path):
                try:
                    shutil.copy2(vid_path, output_dir)
                    success += 1
                except Exception as e:
                    print(f"  ❌ 复制失败 [{Path(vid_path).name}]: {e}")
            else:
                print(f"  ⚠️ 文件不存在: {vid_path}")
        print(f"✅ {label_type} 抽取完成，成功复制 {success} 个。")

    # 执行抽取
    copy_sampled_videos(fn_video_paths_target, fn_output_dir, "FN(漏报)")
    copy_sampled_videos(fp_video_paths_target, fp_output_dir, "FP(误报)")

if __name__ == "__main__":
    PRED_JSON = "/home/ma-user/work/lyf/result1/crash_1cam_2cls_train_3s_39k_0508-800_nothink/result_0509.json" 
    GT_JSONL = "/home/ma-user/work/lyf/data/0506_crash_1cam_2cls_test_39k_3s/test_0506_crash_1cam_2cls_test_39k_3s_front_with_ego_info_5256_3s_clipped_cleaned_dedup_4k.jsonl" 
    
    # 输出目录配置
    FN_SAMPLE_DIR = "/home/ma-user/work/lyf/temp_fn_sample_3s"
    FP_SAMPLE_DIR = "/home/ma-user/work/lyf/temp_fp_sample_3s" 
    
    SAMPLE_NUM = 0
    TARGET_THRESHOLD = 0.5 
    
    calculate_metrics_with_logits(
        PRED_JSON, 
        GT_JSONL, 
        fn_output_dir=FN_SAMPLE_DIR, 
        fp_output_dir=FP_SAMPLE_DIR, 
        sample_num=SAMPLE_NUM, 
        target_threshold=TARGET_THRESHOLD
    )