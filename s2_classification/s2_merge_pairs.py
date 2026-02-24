#!/usr/bin/env python3
"""
s2_merge_pairs.py
CTFAD Pipeline — Step B: 合并多书图文对

功能：
  - 读取 book_001~003 的 image_text_pairs_edited.jsonl
  - 统一 book_id 格式（book_003_t → b003）
  - 生成全局唯一 global_pair_id（如 b001_0042）
  - 将 image_path 转为绝对路径并验证文件是否存在
  - 输出 merged_pairs.jsonl 到 06_classified/

用法：
  conda activate mineru
  python s2_merge_pairs.py

输出：
  /Volumes/aiworkbench/datasets/06_classified/merged_pairs.jsonl
  /Volumes/aiworkbench/datasets/06_classified/merge_report.json
"""

import json
import os
from pathlib import Path
from datetime import datetime

# ============================================================
# 配置（如果路径不同请修改这里）
# ============================================================

DATASETS_ROOT = Path("/Volumes/aiworkbench/datasets")

# 三本书的实际目录名 → 统一 book_id 的映射
BOOK_MAP = {
    "book_001": "b001",
    "book_002": "b002",
    "book_003_t": "b003",   # _t 是历史遗留，统一记为 b003
}

INPUT_FILENAME = "image_text_pairs_edited.jsonl"
OUTPUT_DIR = DATASETS_ROOT / "06_classified"
OUTPUT_FILE = OUTPUT_DIR / "merged_pairs.jsonl"
REPORT_FILE = OUTPUT_DIR / "merge_report.json"

# ============================================================
# 主逻辑
# ============================================================

def resolve_image_path(raw_path: str, book_dir: Path) -> dict:
    """
    将 jsonl 中的相对路径转为绝对路径，并检查文件是否存在。
    raw_path 示例："images/corrected/img_p0000_000.jpg"
    book_dir 示例：/Volumes/.../04_structured/book_001
    """
    abs_path = book_dir / raw_path
    return {
        "absolute_path": str(abs_path),
        "relative_path": raw_path,
        "exists": abs_path.exists()
    }


def merge_pairs():
    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_pairs = []
    report = {
        "created_at": datetime.now().isoformat(),
        "books": {},
        "total_pairs": 0,
        "missing_images": 0,
        "empty_caption_and_context": 0,
    }

    global_counter = 0

    for dir_name, book_id in BOOK_MAP.items():
        book_dir = DATASETS_ROOT / "04_structured" / dir_name
        input_file = book_dir / INPUT_FILENAME

        if not input_file.exists():
            print(f"⚠️  找不到文件：{input_file}，跳过")
            report["books"][book_id] = {"status": "file_not_found", "count": 0}
            continue

        book_pairs = []
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)

                # 生成全局 ID
                global_pair_id = f"{book_id}_{global_counter:04d}"
                global_counter += 1

                # 处理图片路径
                raw_image_path = record.get("image", {}).get("path", "")
                image_info = resolve_image_path(raw_image_path, book_dir)

                # 构建新记录（保留原始字段，追加新字段）
                new_record = {
                    # --- 新增的全局字段 ---
                    "global_pair_id": global_pair_id,
                    "book_id": book_id,
                    "source_dir": dir_name,

                    # --- 原始字段保留 ---
                    "original_pair_id": record.get("pair_id", ""),
                    "image": {
                        "absolute_path": image_info["absolute_path"],
                        "relative_path": image_info["relative_path"],
                        "exists": image_info["exists"],
                        "page": record.get("image", {}).get("page", -1),
                    },
                    "caption": record.get("caption", {}),
                    "context": record.get("context", {}),
                    "alignment": record.get("alignment", {}),
                    "human_edited": record.get("human_edited", False),
                    "review_notes": record.get("review_notes", []),

                    # --- 分类字段（S2 后续填充）---
                    "coarse_label": None,        # Round 1 粗分类
                    "fine_label": None,          # Round 2 细分类
                    "label_confidence": None,
                    "review_status": "pending",  # pending / approved / rejected / corrected
                    "domain_tags": {},           # 领域标签（S2 后期）
                }

                book_pairs.append(new_record)

            # 统计
            missing = sum(1 for p in book_pairs if not p["image"]["exists"])
            empty = sum(
                1 for p in book_pairs
                if not p["caption"].get("text", "").strip()
                and not p["context"].get("text", "").strip()
            )

            report["books"][book_id] = {
                "status": "ok",
                "source_dir": dir_name,
                "count": len(book_pairs),
                "missing_images": missing,
                "empty_caption_and_context": empty,
            }
            report["missing_images"] += missing
            report["empty_caption_and_context"] += empty

            print(f"✅ {dir_name} ({book_id}): {len(book_pairs)} 条，缺图 {missing} 张，空文本 {empty} 条")
            all_pairs.extend(book_pairs)

    # 写入合并文件
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    report["total_pairs"] = len(all_pairs)

    # 写入报告
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"✅ 合并完成：共 {len(all_pairs)} 条图文对")
    print(f"   缺失图片：{report['missing_images']} 张")
    print(f"   空 caption+context：{report['empty_caption_and_context']} 条")
    print(f"   输出文件：{OUTPUT_FILE}")
    print(f"   报告文件：{REPORT_FILE}")
    print(f"{'='*50}")


if __name__ == "__main__":
    merge_pairs()
