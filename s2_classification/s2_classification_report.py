#!/usr/bin/env python3
"""
s2_classification_report.py  v2
CTFAD Pipeline — 分类结果报告生成器

修复说明：
  - 直接读取 fine_label（已是人工审核后的最终标签）
  - original_fine_label 仅用于统计纠错数量
  - 粗分类通过完整映射表推导，不再用字符串分割

用法：
  conda activate mineru
  python s2_classification_report.py
  python s2_classification_report.py --input /path/to/classified_final_fixed.jsonl
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

DATASETS_ROOT = Path("/Volumes/aiworkbench/datasets")
DEFAULT_INPUT = DATASETS_ROOT / "06_classified" / "reviewed" / "classified_final.jsonl"
REPORT_DIR    = DATASETS_ROOT / "06_classified" / "reports"

# ── 完整的细分类 → 粗分类映射 ──
FINE_TO_COARSE = {
    "arrangement_photo":    "arrangement",
    "arrangement_drawing":  "arrangement",
    "arrangement_painting": "arrangement",
    "arrangement":          "arrangement",
    "vessel_photo":         "vessel",
    "vessel_drawing":       "vessel",
    "vessel_painting":      "vessel",
    "vessel":               "vessel",
    "plants_photo":         "plants",
    "plants_drawing":       "plants",
    "plants_painting":      "plants",
    "plants":               "plants",
    "decoration":           "photograph",
    "scene_photo":          "photograph",
    "historical_artifact":  "photograph",
    "photograph":           "photograph",
    "step_diagram":         "diagram",
    "layout_diagram":       "diagram",
    "structure_diagram":    "diagram",
    "diagram":              "diagram",
    "other":                "other",
    "cover_deco":           "other",
    "author_photo":         "other",
    "text_only":            "other",
    "unclassified":         "other",
}

CATEGORY_TREE = {
    "arrangement": {
        "display": "插花作品图",
        "children": {
            "arrangement_photo":    "实拍照片",
            "arrangement_drawing":  "手绘/线描/速写",
            "arrangement_painting": "古画/传统绘画",
            "arrangement":          "未细分",
        }
    },
    "vessel": {
        "display": "花器图",
        "children": {
            "vessel_photo":    "实拍照片",
            "vessel_drawing":  "手绘/线稿",
            "vessel_painting": "传统绘画",
            "vessel":          "未细分",
        }
    },
    "plants": {
        "display": "花材/植物图",
        "children": {
            "plants_photo":    "实拍照片",
            "plants_drawing":  "手绘/线稿",
            "plants_painting": "传统绘画",
            "plants":          "未细分",
        }
    },
    "photograph": {
        "display": "其他实拍照片",
        "children": {
            "decoration":          "装饰物/配饰/案几",
            "scene_photo":         "场景/环境照",
            "historical_artifact": "历史文物/考古实物",
            "photograph":          "未细分",
        }
    },
    "diagram": {
        "display": "示意图/技法图",
        "children": {
            "step_diagram":      "操作步骤图",
            "layout_diagram":    "构图/比例示意图",
            "structure_diagram": "结构示意图",
            "diagram":           "未细分",
        }
    },
    "other": {
        "display": "无法归类",
        "children": {
            "other":        "无法归类",
            "cover_deco":   "封面装饰",
            "author_photo": "作者照",
            "text_only":    "纯文字/表格",
            "unclassified": "真实无法归类",
        }
    },
}

COLLECTION_ADVICE = {
    "arrangement_photo":    "优先级★★★ — 3D生成核心。建议：背景简洁、主体完整、清晰度高",
    "arrangement_drawing":  "优先级★★★ — 草图转3D输入。建议：覆盖多种笔法和风格",
    "arrangement_painting": "优先级★★  — 古画资料。建议：扫描更多宋明清花卉图谱、博古图",
    "arrangement":          "优先级★★  — 未细分条目，建议用 relabel_interface.html 补充细分标注",
    "vessel_photo":         "优先级★★  — 六大器型覆盖度。建议：盘/缸/筒/瓶/篮/碗各类均有",
    "vessel_drawing":       "优先级★   — 器型线稿",
    "vessel_painting":      "优先级★   — 古画中的花器",
    "plants_photo":         "优先级★★  — 花材品种多样性。建议：枝叶花果根各类均有",
    "plants_drawing":       "优先级★   — 花材手绘图鉴",
    "plants_painting":      "优先级★   — 传统植物画",
    "decoration":           "优先级★   — 配饰陈设参考",
    "scene_photo":          "优先级★   — 空间布置参考",
    "historical_artifact":  "优先级★   — 历史考证资料",
    "step_diagram":         "优先级★★  — 技法步骤数据。建议：步骤序列尽量完整",
    "layout_diagram":       "优先级★★  — 3D构图参数来源。建议：含角度数值、比例标注",
    "structure_diagram":    "优先级★   — 结构说明图",
    "other":                "优先级—   — 尽量减少此类别",
    "unclassified":         "优先级—   — 真实无法归类，正常保留少量",
}


def load_records(path: Path) -> list:
    recs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def compute_stats(records: list) -> dict:
    total       = len(records)
    fine_dist   = defaultdict(int)
    coarse_dist = defaultdict(int)
    by_book     = defaultdict(lambda: defaultdict(int))
    source_dist = defaultdict(int)

    # 纠错数：is_corrected=True 且 fine_label != original_fine_label
    corrected_count = sum(
        1 for r in records
        if r.get("is_corrected", False)
        and r.get("fine_label") != r.get("original_fine_label")
    )

    for r in records:
        fine   = r.get("fine_label") or "unclassified"
        book   = r.get("book_id", "unknown")
        source = r.get("label_source", "unknown")

        fine_dist[fine]    += 1
        source_dist[source] += 1
        by_book[book][fine] += 1
        coarse_dist[FINE_TO_COARSE.get(fine, "other")] += 1

    return {
        "total":           total,
        "fine_dist":       dict(fine_dist),
        "coarse_dist":     dict(coarse_dist),
        "by_book":         {k: dict(v) for k, v in by_book.items()},
        "corrected_count": corrected_count,
        "source_dist":     dict(source_dist),
    }


def generate_markdown(stats: dict, input_file: Path) -> str:
    total   = stats["total"]
    fine    = stats["fine_dist"]
    coarse  = stats["coarse_dist"]
    by_book = stats["by_book"]
    books   = sorted(by_book.keys())

    lines = [
        "# CTFAD 数据集分类报告",
        "",
        f"**生成时间：** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**数据来源：** `{input_file.name}`",
        f"**总图文对：** {total} 条",
        f"**涵盖书目：** {', '.join(books)}（共 {len(books)} 本）",
        f"**人工纠错：** {stats['corrected_count']} 条"
        f"（{stats['corrected_count']/total*100:.1f}%）",
        "",
        "---",
        "",
        "## 一、粗分类汇总",
        "",
        "| 大类 | 数量 | 占比 | 条形图 |",
        "|---|---|---|---|",
    ]

    for key, info in CATEGORY_TREE.items():
        cnt = coarse.get(key, 0)
        pct = cnt / total * 100 if total else 0
        bar = "█" * int(pct / 2.5)
        lines.append(f"| **{info['display']}** (`{key}`) | {cnt} | {pct:.1f}% | {bar} |")

    lines += ["", "---", "", "## 二、细分类详情", ""]

    for key, info in CATEGORY_TREE.items():
        top_cnt = coarse.get(key, 0)
        top_pct = top_cnt / total * 100 if total else 0
        lines += [
            f"### {info['display']}（`{key}`）— 共 {top_cnt} 条（{top_pct:.1f}%）",
            "",
            "| 细分类 | 含义 | 数量 | 占大类% | 占总量% |",
            "|---|---|---|---|---|",
        ]
        for fk, fd in info["children"].items():
            cnt = fine.get(fk, 0)
            if cnt == 0:
                continue
            lines.append(
                f"| `{fk}` | {fd} | {cnt} "
                f"| {cnt/top_cnt*100:.0f}% | {cnt/total*100:.1f}% |"
            )
        lines.append("")

    # 按书目
    lines += [
        "---", "",
        "## 三、按书目分布", "",
        "| 细分类 | " + " | ".join(books) + " | 合计 |",
        "|---" * (len(books) + 2) + "|",
    ]
    for fl in sorted(fine.keys(), key=lambda x: (FINE_TO_COARSE.get(x, "z"), x)):
        row = f"| `{fl}` |"
        total_fl = 0
        for bk in books:
            cnt = by_book[bk].get(fl, 0)
            total_fl += cnt
            row += f" {cnt} |"
        row += f" **{total_fl}** |"
        lines.append(row)

    # 采集建议
    lines += [
        "", "---", "",
        "## 四、数据采集建议", "",
        "| 细分类 | 当前数量 | 建议 |",
        "|---|---|---|",
    ]
    for fl, advice in COLLECTION_ADVICE.items():
        cnt = fine.get(fl, 0)
        if cnt > 0:
            lines.append(f"| `{fl}` | {cnt}（{cnt/total*100:.1f}%）| {advice} |")

    # 质量
    lines += [
        "", "---", "",
        "## 五、数据质量概览", "",
        "| 指标 | 数值 |",
        "|---|---|",
        f"| 总图文对 | {total} |",
        f"| 书目数量 | {len(books)} |",
        f"| 人工纠错 | {stats['corrected_count']}（{stats['corrected_count']/total*100:.1f}%）|",
    ]
    for src, cnt in sorted(stats["source_dist"].items(), key=lambda x: -x[1]):
        label = {"human_sandbox": "沙盒人工标注",
                 "model_round2": "Round 2 模型预测",
                 "model_round1_only": "Round 1 粗分类"}.get(src, src)
        lines.append(f"| 来源：{label} | {cnt} |")

    lines += ["", "---",
              "*本报告由 `s2_classification_report.py` v2 自动生成，基于人工审核后的最终标签。*",
              "*每次新增数据后重新运行即可更新。*"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    args = parser.parse_args()

    input_file = Path(args.input)
    if not input_file.exists():
        print(f"❌ 找不到：{input_file}")
        return

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")

    print(f"📊 CTFAD 分类报告生成器 v2")
    print(f"   输入：{input_file}\n")

    recs  = load_records(input_file)
    stats = compute_stats(recs)

    print(f"   粗分类分布（人工审核后最终标签）：")
    for key, info in CATEGORY_TREE.items():
        cnt = stats["coarse_dist"].get(key, 0)
        pct = cnt / stats["total"] * 100 if stats["total"] else 0
        bar = "█" * int(pct / 5)
        print(f"   {info['display']:12s} {cnt:4d} ({pct:5.1f}%) {bar}")

    print(f"\n   细分类：")
    for fl, cnt in sorted(stats["fine_dist"].items(), key=lambda x: -x[1]):
        pct = cnt / stats["total"] * 100
        print(f"   {fl:30s} {cnt:4d} ({pct:4.1f}%)")

    md = generate_markdown(stats, input_file)
    (REPORT_DIR / f"classification_report_{ts}.md").write_text(md, encoding="utf-8")
    (REPORT_DIR / "classification_report_latest.md").write_text(md, encoding="utf-8")
    (REPORT_DIR / f"classification_report_{ts}.json").write_text(
        json.dumps({"generated_at": datetime.now().isoformat(),
                    "input_file": str(input_file), "stats": stats},
                   ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"\n✅ 报告已生成")
    print(f"   {REPORT_DIR}/classification_report_{ts}.md")
    print(f"   {REPORT_DIR}/classification_report_latest.md")


if __name__ == "__main__":
    main()
