# -*- coding: utf-8 -*-
"""碰撞风险二分类 GRPO 自定义 reward。

verl 通过 reward.custom_reward_function.path / .name 加载本文件的 compute_score,
调用签名固定为:
    compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs)

任务约束为无 CoT 直接输出标签("高风险" / "安全"), 且模型已经过 SFT 学会输出
裸标签, 因此奖励采用**严格精确匹配** (仅容忍首尾空白与标点):
    - 剥掉首尾空白/标点后, 输出必须恰好等于某个合法标签才算有效预测;
    - 预测 == ground_truth  -> score 1.0, 否则 0.0。

不做子串提取的原因: "不安全"/"非高风险"/"存在安全隐患" 等否定或复合表述
包含标签子串, 子串匹配会把语义相反的输出判满分, 形成可被 GRPO 强化的
reward hacking 漏洞; 严格匹配同时也把奖励与"格式正确"绑定, 与无 CoT 约束一致。
"""

LABELS = ("高风险", "安全")
_STRIP_CHARS = " \t\r\n。.!！?？,，;；:：\"'“”‘’"


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    gt = str(ground_truth).strip()
    solution = (solution_str or "").strip().strip(_STRIP_CHARS)

    valid = solution in LABELS
    acc = 1.0 if (valid and solution == gt) else 0.0

    return {
        "score": acc,
        "acc": acc,
        "format": 1.0 if valid else 0.0,
        "pred": solution if valid else "none",
    }


if __name__ == "__main__":
    # 简单自测
    cases = [
        ("高风险", "高风险", 1.0),
        ("安全", "安全", 1.0),
        ("高风险", "安全", 0.0),
        ("安全。", "安全", 1.0),             # 仅容忍首尾标点
        (" 高风险\n", "高风险", 1.0),
        ("我认为是高风险", "高风险", 0.0),   # 冗余前缀: 严格匹配不给分
        ("不安全", "安全", 0.0),             # 否定表述: 不能因子串得分
        ("非高风险", "高风险", 0.0),
        ("存在安全隐患", "安全", 0.0),
        ("", "高风险", 0.0),
        ("不确定", "安全", 0.0),
        ("安全还是高风险呢", "高风险", 0.0),
    ]
    for sol, gt, want in cases:
        got = compute_score("collision_risk", sol, gt)
        assert got["score"] == want, (sol, gt, got, want)
        print(f"OK: {sol!r} vs {gt!r} -> {got}")
    print("all reward self-tests passed")
