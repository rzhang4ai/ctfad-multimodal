#!/usr/bin/env python3
"""
s2_merge_results.py
CTFAD Pipeline — 合并所有分类结果，生成审核就绪文件

数据来源优先级：
  1. 沙盒人工标注（human_sandbox）→ 最高优先级，直接用
  2. Round 2 模型细分类（model_round2）→ 正常使用
  3. Round 1 兜底（model_round1_only）→ 仅 other 类

输出字段含义：
  fine_label       最终细分类标签
  label_source     标签来源
  review_tier      审核档位：skip / batch / careful
    skip    = 沙盒已人工标注，不需要审核
    batch   = 高准确率类别（arrangement/diagram/plants），进入批量确认区
    careful = 低准确率类别（vessel/photograph/other），逐条审核

用法：
  conda activate mineru
  python s2_merge_results.py

输出：
  /Volumes/aiworkbench/datasets/06_classified/review_ready.jsonl
  /Volumes/aiworkbench/datasets/06_classified/merge_summary.json
"""

import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

DATASETS_ROOT   = Path("/Volumes/aiworkbench/datasets")
ROUND2_FILE     = DATASETS_ROOT / "06_classified" / "round2_predictions.jsonl"
SANDBOX_FILE    = DATASETS_ROOT / "06_classified" / "sandbox" / "sandbox_reviewed.jsonl"
MERGED_FILE     = DATASETS_ROOT / "06_classified" / "merged_pairs.jsonl"
OUTPUT_FILE     = DATASETS_ROOT / "06_classified" / "review_ready.jsonl"
SUMMARY_FILE    = DATASETS_ROOT / "06_classified" / "merge_summary.json"

# 类别准确率驱动的审核档位
# batch   = 沙盒准确率 >= 80%，进入批量确认区
# careful = 沙盒准确率 < 80%，逐条审核
BATCH_CATEGORIES   = {"arrangement", "diagram", "plants"}
CAREFUL_CATEGORIES = {"vessel", "photograph", "other"}


def main():
    print("🌸 CTFAD 合并分类结果\n")

    # ── 读取沙盒人工标注 ──
    sandbox_human = {}
    if SANDBOX_FILE.exists():
        with open(SANDBOX_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                gid = r.get("global_pair_id")
                if gid and r.get("review_status") != "pending":
                    sandbox_human[gid] = {
                        "human_label":  r.get("human_label"),
                        "is_correct":   r.get("is_correct"),
                        "predicted":    r.get("predicted_label"),
                    }
    print(f"   沙盒人工标注：{len(sandbox_human)} 条")

    # ── 读取 Round 2 结果 ──
    round2 = {}
    with open(ROUND2_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                round2[r["global_pair_id"]] = r
    print(f"   Round 2 结果：{len(round2)} 条")

    # ── 读取原始合并数据（获取完整图文信息）──
    merged = {}
    with open(MERGED_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                merged[r["global_pair_id"]] = r
    print(f"   原始图文对：{len(merged)} 条\n")

    # ── 合并 ──
    results = []
    stats = defaultdict(int)

    for gid, pair in merged.items():
        r2 = round2.get(gid, {})
        sandbox = sandbox_human.get(gid)

        coarse_label     = r2.get("coarse_label", "other")
        coarse_confidence = r2.get("coarse_confidence", 0.0)
        fine_label       = r2.get("fine_label", coarse_label)
        fine_confidence  = r2.get("fine_confidence", 0.0)
        fine_reason      = r2.get("fine_reason", "")
        label_source     = r2.get("label_source", "model_round2")

        # 沙盒人工标注：覆盖模型结果
        if sandbox:
            fine_label   = sandbox["human_label"]
            label_source = "human_sandbox"
            review_tier  = "skip"
            stats["skip"] += 1
        # 高准确率类别 → 批量确认区
        elif coarse_label in BATCH_CATEGORIES:
            review_tier = "batch"
            stats["batch"] += 1
        # 低准确率类别 → 逐条审核区
        else:
            review_tier = "careful"
            stats["careful"] += 1

        caption = pair.get("caption", {}).get("text", "") or ""
        context = pair.get("context", {}).get("text", "") or ""

        record = {
            "global_pair_id":     gid,
            "book_id":            pair.get("book_id", ""),
            "image_path":         pair["image"]["absolute_path"],
            "image_filename":     Path(pair["image"]["absolute_path"]).name,
            "page":               pair["image"].get("page", -1),
            "caption":            caption,
            "context":            context[:300],
            "coarse_label":       coarse_label,
            "coarse_confidence":  coarse_confidence,
            "fine_label":         fine_label,
            "fine_confidence":    fine_confidence,
            "fine_reason":        fine_reason,
            "label_source":       label_source,
            "review_tier":        review_tier,
            # 审核结果字段（由审核界面填写）
            "review_status":      "pending" if review_tier != "skip" else "approved",
            "human_fine_label":   fine_label if review_tier == "skip" else None,
            "reviewed_at":        datetime.now().isoformat() if review_tier == "skip" else None,
        }
        results.append(record)

    # ── 写出 ──
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── 汇总 ──
    fine_dist = defaultdict(int)
    for r in results:
        fine_dist[r["fine_label"]] += 1

    summary = {
        "created_at":   datetime.now().isoformat(),
        "total":        len(results),
        "review_tiers": dict(stats),
        "fine_label_distribution": dict(sorted(fine_dist.items(), key=lambda x: -x[1])),
    }
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"{'='*50}")
    print(f"✅ 合并完成：共 {len(results)} 条")
    print(f"\n   审核档位分布：")
    print(f"   ⏭  skip（沙盒已标注，直接跳过）：{stats['skip']} 条")
    print(f"   🔲 batch（批量确认区）：{stats['batch']} 条")
    print(f"   🔍 careful（逐条审核区）：{stats['careful']} 条")
    print(f"\n   细分类分布：")
    for label, count in sorted(fine_dist.items(), key=lambda x: -x[1]):
        print(f"   {label}: {count}")
    print(f"\n   输出：{OUTPUT_FILE}")
    print(f"   下一步：用 review_interface.html 审核")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
