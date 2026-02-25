#!/usr/bin/env python3
"""
s2_calibrate_confidence.py
CTFAD Pipeline — 置信度校准

功能：
  - 读取沙盒人工审核结果（sandbox_reviewed.jsonl）
  - 分析每个置信度区间的实际准确率
  - 结合全量预测数据的置信度分布
  - 输出推荐的高/中/低置信度阈值
  - 生成校准报告

用法：
  conda activate mineru
  python s2_calibrate_confidence.py

输出：
  /Volumes/aiworkbench/datasets/06_classified/confidence_calibration.json
  /Volumes/aiworkbench/datasets/06_classified/confidence_calibration_report.md
"""

import json
from pathlib import Path
from collections import defaultdict

DATASETS_ROOT  = Path("/Volumes/aiworkbench/datasets")
SANDBOX_FILE   = DATASETS_ROOT / "06_classified" / "sandbox" / "sandbox_reviewed.jsonl"
FULL_PRED_FILE = DATASETS_ROOT / "06_classified" / "full_predictions.jsonl"
OUT_JSON       = DATASETS_ROOT / "06_classified" / "confidence_calibration.json"
OUT_REPORT     = DATASETS_ROOT / "06_classified" / "confidence_calibration_report.md"

# 置信度区间划分（用于分析）
BINS = [
    (0.9, 1.01, "0.90-1.00"),
    (0.8, 0.9,  "0.80-0.89"),
    (0.7, 0.8,  "0.70-0.79"),
    (0.6, 0.7,  "0.60-0.69"),
    (0.0, 0.6,  "0.00-0.59"),
]


def get_bin(conf):
    for lo, hi, label in BINS:
        if lo <= conf < hi:
            return label
    return "0.00-0.59"


def main():
    # ── 读取沙盒审核结果 ──
    sandbox = []
    with open(SANDBOX_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                if r.get("review_status") != "pending":
                    sandbox.append(r)

    print(f"📊 沙盒已审核样本：{len(sandbox)} 条")

    # ── 按置信度区间统计准确率 ──
    bin_stats = defaultdict(lambda: {"total": 0, "correct": 0, "records": []})

    for r in sandbox:
        conf = r.get("predicted_confidence") or 0.0
        is_correct = r.get("is_correct", False)
        b = get_bin(conf)
        bin_stats[b]["total"] += 1
        bin_stats[b]["records"].append(conf)
        if is_correct:
            bin_stats[b]["correct"] += 1

    print("\n   置信度区间 → 实际准确率（基于沙盒）：")
    bin_accuracy = {}
    for _, _, label in BINS:
        stats = bin_stats[label]
        if stats["total"] > 0:
            acc = stats["correct"] / stats["total"]
            bin_accuracy[label] = acc
            print(f"   {label}：{acc*100:.0f}%  ({stats['correct']}/{stats['total']})")
        else:
            bin_accuracy[label] = None
            print(f"   {label}：— （无样本）")

    # ── 读取全量预测分布 ──
    full_conf_dist = defaultdict(int)
    full_total = 0
    with open(FULL_PRED_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            conf = r.get("prediction", {}).get("confidence") or 0.0
            full_conf_dist[get_bin(conf)] += 1
            full_total += 1

    print(f"\n   全量数据（{full_total}条）置信度分布：")
    for _, _, label in BINS:
        count = full_conf_dist[label]
        pct = count / full_total * 100 if full_total > 0 else 0
        print(f"   {label}：{count} 条 ({pct:.0f}%)")

    # ── 推导推荐阈值 ──
    # 逻辑：找到实际准确率 >= 90% 的最低置信度区间下界 → 作为 HIGH 阈值
    #       找到实际准确率 >= 75% 的最低置信度区间下界 → 作为 MEDIUM 阈值
    #       低于 MEDIUM 阈值 → LOW（必须逐条审核）

    high_threshold  = None
    medium_threshold = None

    for lo, hi, label in BINS:  # 从高到低遍历
        acc = bin_accuracy.get(label)
        if acc is None:
            continue
        if acc >= 0.90 and high_threshold is None:
            high_threshold = lo
        if acc >= 0.75 and medium_threshold is None:
            medium_threshold = lo

    # 兜底：如果样本不足导致无法推导，使用保守默认值
    if high_threshold is None:
        high_threshold = 0.85
        print("\n   ⚠️  沙盒样本不足，高置信度阈值使用默认值 0.85")
    if medium_threshold is None:
        medium_threshold = 0.70
        print("   ⚠️  沙盒样本不足，中置信度阈值使用默认值 0.70")

    # 统计各档位的全量数量
    high_count   = sum(full_conf_dist[label] for lo, hi, label in BINS if lo >= high_threshold)
    medium_count = sum(full_conf_dist[label] for lo, hi, label in BINS if medium_threshold <= lo < high_threshold)
    low_count    = sum(full_conf_dist[label] for lo, hi, label in BINS if lo < medium_threshold)

    print(f"\n{'='*50}")
    print(f"📐 推荐置信度阈值：")
    print(f"   高置信度（HIGH）  ≥ {high_threshold}  → {high_count} 条（批量确认）")
    print(f"   中置信度（MEDIUM）{medium_threshold}-{high_threshold} → {medium_count} 条（快速扫描）")
    print(f"   低置信度（LOW）   < {medium_threshold}  → {low_count} 条（逐条审核）")
    print(f"{'='*50}")

    # ── 保存结果 ──
    calibration = {
        "high_threshold":   high_threshold,
        "medium_threshold": medium_threshold,
        "bin_accuracy":     bin_accuracy,
        "full_distribution": {label: full_conf_dist[label] for _, _, label in BINS},
        "full_total":       full_total,
        "high_count":       high_count,
        "medium_count":     medium_count,
        "low_count":        low_count,
        "sandbox_size":     len(sandbox),
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(calibration, f, ensure_ascii=False, indent=2)

    # ── 生成报告 ──
    report = f"""# CTFAD 置信度校准报告

**生成时间：** 基于 {len(sandbox)} 条沙盒人工审核样本

---

## 推荐阈值

| 档位 | 置信度范围 | 全量数量 | 操作建议 |
|---|---|---|---|
| 高置信度 | ≥ {high_threshold} | {high_count} 条 | 缩略图扫描，一键批量确认 |
| 中置信度 | {medium_threshold} ~ {high_threshold} | {medium_count} 条 | 快速逐条确认 |
| 低置信度 | < {medium_threshold} | {low_count} 条 | 必须逐条审核 |

---

## 沙盒置信度区间准确率

| 置信度区间 | 实际准确率 | 样本数 |
|---|---|---|
"""
    for _, _, label in BINS:
        acc = bin_accuracy.get(label)
        stats = bin_stats[label]
        acc_str = f"{acc*100:.0f}%" if acc is not None else "—"
        report += f"| {label} | {acc_str} | {stats['total']} |\n"

    report += f"""
---

## 全量数据置信度分布

| 置信度区间 | 数量 | 占比 |
|---|---|---|
"""
    for _, _, label in BINS:
        count = full_conf_dist[label]
        pct = count / full_total * 100 if full_total > 0 else 0
        report += f"| {label} | {count} | {pct:.0f}% |\n"

    report += f"""
---

*注：阈值基于沙盒 {len(sandbox)} 条样本推导，随着人工审核积累可定期重新校准。*
"""

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n✅ 校准完成")
    print(f"   JSON：{OUT_JSON}")
    print(f"   报告：{OUT_REPORT}")
    print(f"\n   下一步：运行 s2_classify_round2.py")


if __name__ == "__main__":
    main()
