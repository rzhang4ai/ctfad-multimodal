#!/usr/bin/env python3
"""
s2_analyze_sandbox.py
CTFAD Pipeline — Step D: 沙盒结果分析 + Prompt 优化建议

功能：
  - 读取人工审核后的 sandbox_reviewed.jsonl
  - 计算整体准确率、各类别准确率
  - 找出最常见的混淆对
  - 生成 prompt 优化建议报告

用法：
  conda activate mineru
  python s2_analyze_sandbox.py

输入：
  /Volumes/aiworkbench/datasets/06_classified/sandbox/sandbox_reviewed.jsonl
输出：
  /Volumes/aiworkbench/datasets/06_classified/sandbox/sandbox_analysis.json
  /Volumes/aiworkbench/datasets/06_classified/sandbox/sandbox_analysis_report.md
"""

import json
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

# ============================================================
# 配置
# ============================================================

SANDBOX_DIR = Path("/Volumes/aiworkbench/datasets/06_classified/sandbox")
REVIEWED_FILE = SANDBOX_DIR / "sandbox_reviewed.jsonl"
ANALYSIS_JSON = SANDBOX_DIR / "sandbox_analysis.json"
ANALYSIS_REPORT = SANDBOX_DIR / "sandbox_analysis_report.md"

# Round 1 粗分类的 top-level 类别
TOP_LEVEL = ["arrangement", "vessel", "plants", "photograph", "diagram", "other"]

# 细分类到粗分类的映射
FINE_TO_COARSE = {
    "arrangement_photo": "arrangement",
    "arrangement_drawing": "arrangement",
    "arrangement_painting": "arrangement",
    "vessel_photo": "vessel",
    "vessel_painting": "vessel",
    "vessel_drawing": "vessel",
    "plants_photo": "plants",
    "plants_painting": "plants",
    "plants_drawing": "plants",
    "decoration": "photograph",
    "scene_photo": "photograph",
    "historical_artifact": "photograph",
    "step_diagram": "diagram",
    "layout_diagram": "diagram",
    "structure_diagram": "diagram",
    "cover_deco": "other",
    "author_photo": "other",
    "text_only": "other",
    "unclassified": "other",
}

# Prompt 优化建议模板（根据混淆对生成）
CONFUSION_ADVICE = {
    ("arrangement", "diagram"): (
        "【arrangement vs diagram 混淆】\n"
        "  问题：模型把插花作品图误判为示意图（或反之）\n"
        "  优化方向：在 prompt 中强调\n"
        "  - diagram 的核心特征是「文字标注、箭头、数字序号、比例线」等说明性元素\n"
        "  - 如果图片是完成的插花作品照片，即使旁边有文字说明，主体仍是 arrangement\n"
        "  - 建议加入 few-shot 示例：一张有箭头的步骤图 vs 一张无标注的作品照"
    ),
    ("arrangement", "plants"): (
        "【arrangement vs plants 混淆】\n"
        "  问题：模型把插花作品误判为花材图（或反之）\n"
        "  优化方向：\n"
        "  - 强调「已完成的插花作品」的判断标准：花材被插在花器中，形成完整作品\n"
        "  - plants 仅用于「单独拍摄的花材素材」，没有花器"
    ),
    ("arrangement", "photograph"): (
        "【arrangement vs photograph 混淆】\n"
        "  问题：模型把插花场景照误判为普通场景照（或反之）\n"
        "  优化方向：\n"
        "  - 强调「插花作品是画面绝对主体」才算 arrangement\n"
        "  - 如果插花只是场景的一小部分，归 scene_photo\n"
        "  - 建议在 prompt 中加：「如果画面中心是插花作品，选 arrangement；如果插花只是背景装饰，选 photograph」"
    ),
    ("arrangement", "vessel"): (
        "【arrangement vs vessel 混淆】\n"
        "  问题：花器特写被误判为插花作品（或反之）\n"
        "  优化方向：\n"
        "  - 强调 vessel 是「空的或未完成的花器」\n"
        "  - 只要花器里有花材构成作品，就是 arrangement"
    ),
    ("diagram", "plants"): (
        "【diagram vs plants 混淆】\n"
        "  问题：花材手绘线稿被误判为步骤图（或反之）\n"
        "  优化方向：\n"
        "  - 强调 diagram 必须有「说明性元素」（箭头、数字、文字标注）\n"
        "  - 纯粹的花材线稿没有标注，应归 plants_drawing"
    ),
    ("arrangement", "other"): (
        "【arrangement vs other 混淆】\n"
        "  问题：模型对插花作品没有信心，归入 other\n"
        "  优化方向：\n"
        "  - 在 prompt 末尾强调：「如果图中有花材和花器组合，应归 arrangement 而非 other」\n"
        "  - 降低 other 的使用门槛说明，让模型更大胆选具体类别"
    ),
}


# ============================================================
# 主分析逻辑
# ============================================================

def normalize_to_coarse(label: str) -> str:
    """将细分类标签统一到粗分类"""
    if label in TOP_LEVEL:
        return label
    return FINE_TO_COARSE.get(label, "other")


def analyze():
    if not REVIEWED_FILE.exists():
        print(f"❌ 找不到审核文件：{REVIEWED_FILE}")
        print("   请先完成 sandbox_review.html 审核，导出 sandbox_reviewed.jsonl 并放入 sandbox/ 目录")
        return

    records = []
    with open(REVIEWED_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    reviewed = [r for r in records if r.get("review_status") != "pending"]
    total = len(records)
    reviewed_count = len(reviewed)

    print(f"📊 载入 {total} 条记录，其中 {reviewed_count} 条已审核")

    if reviewed_count == 0:
        print("⚠️  没有已审核的记录，请先完成审核")
        return

    # ── 准确率计算 ──
    correct_count = sum(1 for r in reviewed if r.get("is_correct"))
    overall_accuracy = correct_count / reviewed_count

    # ── 按粗分类统计 ──
    by_coarse_predicted = defaultdict(list)   # 预测类别 → 记录列表
    by_coarse_human = defaultdict(list)        # 人工类别 → 记录列表
    confusion_pairs = Counter()                # (预测, 人工) → 次数

    for r in reviewed:
        pred_coarse = normalize_to_coarse(r.get("predicted_label", "other"))
        human_label = r.get("human_label") or r.get("predicted_label", "other")
        human_coarse = normalize_to_coarse(human_label)

        by_coarse_predicted[pred_coarse].append(r)
        by_coarse_human[human_coarse].append(r)

        if not r.get("is_correct"):
            confusion_pairs[(pred_coarse, human_coarse)] += 1

    # ── 各类别准确率 ──
    category_stats = {}
    for cat in TOP_LEVEL:
        cat_records = [r for r in reviewed if normalize_to_coarse(r.get("predicted_label", "other")) == cat]
        if cat_records:
            cat_correct = sum(1 for r in cat_records if r.get("is_correct"))
            category_stats[cat] = {
                "predicted_count": len(cat_records),
                "correct": cat_correct,
                "accuracy": cat_correct / len(cat_records),
            }
        else:
            category_stats[cat] = {"predicted_count": 0, "correct": 0, "accuracy": None}

    # ── 真实分布（人工标注）──
    human_distribution = {
        cat: len([r for r in reviewed if normalize_to_coarse(r.get("human_label") or r.get("predicted_label", "")) == cat])
        for cat in TOP_LEVEL
    }

    # ── 混淆矩阵（Top 5 错误对）──
    top_confusions = confusion_pairs.most_common(5)

    # ── 组装分析结果 ──
    analysis = {
        "analyzed_at": datetime.now().isoformat(),
        "total_records": total,
        "reviewed_count": reviewed_count,
        "overall_accuracy": round(overall_accuracy, 4),
        "correct_count": correct_count,
        "wrong_count": reviewed_count - correct_count,
        "category_stats": category_stats,
        "human_distribution": human_distribution,
        "top_confusions": [
            {"predicted": p, "human_corrected": h, "count": c}
            for (p, h), c in top_confusions
        ],
    }

    # 保存 JSON
    with open(ANALYSIS_JSON, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)

    # ── 生成 Markdown 报告 ──
    report_lines = [
        "# CTFAD 沙盒分类分析报告",
        f"\n**生成时间：** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**审核样本：** {reviewed_count} / {total} 条",
        "",
        "---",
        "",
        "## 一、总体准确率",
        "",
        f"| 指标 | 数值 |",
        f"|---|---|",
        f"| 总体准确率 | **{overall_accuracy*100:.1f}%** ({correct_count}/{reviewed_count}) |",
        f"| 错误数 | {reviewed_count - correct_count} 条 |",
        "",
        "**准确率评估标准：**",
        "- ≥ 85%：可直接全量运行",
        "- 70-85%：优化 prompt 后再全量运行",
        "- < 70%：考虑换更大模型或改用云端 API",
        "",
        "---",
        "",
        "## 二、各类别准确率",
        "",
        "| 类别 | 预测次数 | 正确 | 准确率 |",
        "|---|---|---|---|",
    ]

    for cat, stats in category_stats.items():
        if stats["predicted_count"] > 0:
            acc_str = f"{stats['accuracy']*100:.0f}%"
            flag = " ⚠️" if stats["accuracy"] is not None and stats["accuracy"] < 0.7 else ""
            report_lines.append(f"| `{cat}` | {stats['predicted_count']} | {stats['correct']} | {acc_str}{flag} |")
        else:
            report_lines.append(f"| `{cat}` | 0 | — | — |")

    report_lines += [
        "",
        "---",
        "",
        "## 三、真实类别分布（人工标注）",
        "",
        "| 类别 | 数量 | 占比 |",
        "|---|---|---|",
    ]
    for cat, count in sorted(human_distribution.items(), key=lambda x: -x[1]):
        pct = count / reviewed_count * 100 if reviewed_count > 0 else 0
        report_lines.append(f"| `{cat}` | {count} | {pct:.1f}% |")

    report_lines += [
        "",
        "---",
        "",
        "## 四、主要混淆对（需重点优化）",
        "",
    ]

    if top_confusions:
        report_lines += [
            "| 模型预测 | 正确类别 | 次数 |",
            "|---|---|---|",
        ]
        for (pred, human), count in top_confusions:
            report_lines.append(f"| `{pred}` | `{human}` | {count} |")

        report_lines += ["", "### Prompt 优化建议", ""]

        seen_pairs = set()
        for (pred, human), count in top_confusions:
            pair = tuple(sorted([pred, human]))
            if pair not in seen_pairs and pair in CONFUSION_ADVICE:
                seen_pairs.add(pair)
                report_lines.append(CONFUSION_ADVICE[pair])
                report_lines.append("")
    else:
        report_lines.append("🎉 没有发现主要混淆对，模型表现良好！")

    report_lines += [
        "",
        "---",
        "",
        "## 五、下一步建议",
        "",
    ]

    if overall_accuracy >= 0.85:
        report_lines += [
            "✅ **准确率达标，可直接运行全量分类：**",
            "```bash",
            "python s2_classify_full.py",
            "```",
        ]
    elif overall_accuracy >= 0.70:
        report_lines += [
            "⚠️  **建议先优化 Prompt，再全量运行：**",
            "1. 根据上方「Prompt 优化建议」修改 `s2_classify_sandbox.py` 中的 `build_classification_prompt()`",
            "2. 删除 `sandbox_predictions.jsonl`，重新跑 40 张验证",
            "3. 确认准确率 ≥ 85% 后再全量运行",
        ]
    else:
        report_lines += [
            "❌ **准确率不足，建议：**",
            "1. 先尝试优化 Prompt（见上方建议）",
            "2. 如果优化后仍 < 70%，考虑使用云端 API（Qwen-VL-Max 或 GPT-4o）",
            "3. 可在 `model_config.yaml` 中切换模型后重新测试",
        ]

    report_text = "\n".join(report_lines)
    with open(ANALYSIS_REPORT, "w", encoding="utf-8") as f:
        f.write(report_text)

    # ── 终端输出摘要 ──
    print(f"\n{'='*50}")
    print(f"📊 分析完成")
    print(f"   总体准确率：{overall_accuracy*100:.1f}% ({correct_count}/{reviewed_count})")
    print(f"\n   各类别准确率：")
    for cat, stats in category_stats.items():
        if stats["predicted_count"] > 0:
            flag = " ⚠️ " if stats["accuracy"] is not None and stats["accuracy"] < 0.7 else "    "
            print(f"   {flag}{cat}: {stats['accuracy']*100:.0f}% ({stats['correct']}/{stats['predicted_count']})")

    if top_confusions:
        print(f"\n   主要混淆对：")
        for (pred, human), count in top_confusions[:3]:
            print(f"     {pred} → 应为 {human}（{count}次）")

    print(f"\n   详细报告：{ANALYSIS_REPORT}")
    print(f"{'='*50}")

    if overall_accuracy >= 0.85:
        print("\n✅ 准确率达标，可以运行全量分类")
    elif overall_accuracy >= 0.70:
        print("\n⚠️  建议先优化 Prompt，再全量运行")
    else:
        print("\n❌ 准确率不足，建议优化 Prompt 或换模型")


if __name__ == "__main__":
    analyze()
