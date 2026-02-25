#!/usr/bin/env python3
"""
s2_classify_sandbox.py
CTFAD Pipeline — Step D: 沙盒分类测试

功能：
  - 从 merged_pairs.jsonl 中按策略抽取 40 张图（覆盖三本书）
  - 调用本地 qwen2.5vl:3b 进行 Round 1 粗分类
  - 输出预测结果 sandbox_predictions.jsonl
  - 遵循三条铁律：sleep 冷却 + 断点续传 + 图片预压缩

用法：
  conda activate mineru
  python s2_classify_sandbox.py

输出：
  /Volumes/aiworkbench/datasets/06_classified/sandbox/sandbox_predictions.jsonl
"""

import json
import os
import time
import base64
import random
from pathlib import Path
from datetime import datetime
from io import BytesIO

import requests
from PIL import Image

# ============================================================
# 配置
# ============================================================

DATASETS_ROOT = Path("/Volumes/aiworkbench/datasets")
MERGED_FILE = DATASETS_ROOT / "06_classified" / "merged_pairs.jsonl"
SANDBOX_DIR = DATASETS_ROOT / "06_classified" / "sandbox"
OUTPUT_FILE = SANDBOX_DIR / "sandbox_predictions.jsonl"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5vl:3b"

SAMPLE_SIZE = 40          # 抽样数量
IMAGE_MAX_SIZE = 768      # 图片压缩最大边长（铁律2）
SLEEP_BETWEEN = 2.5       # 每次调用后冷却（铁律1）
SLEEP_EVERY_N = 20        # 每N次额外冷却
SLEEP_EXTRA = 10          # 额外冷却秒数

# Round 1 粗分类的类别（对应 image_text_categories.yaml 的 top-level）
COARSE_CATEGORIES = {
    "arrangement": "插花作品图——画面主体是已完成的插花作品（含速写、手绘、古画等风格），无论实拍/手绘/古画",
    "vessel": "花器图——画面主体是单独的空花器/容器，不含任何花材，也不是操作步骤",
    "plants": "花材/植物图——画面主体是单独的花材、植物素材，没有花器",
    "photograph": "其他实拍照片——场景、装饰物、文物等，不含插花作品",
    "diagram": "示意图/步骤图——操作步骤图（即使无箭头）、构图比例图、结构说明图；图注含「步骤」「步骤N」字样的优先归此类",
    "other": "无法归类——封面装饰、作者照、模糊图等，确实无法归入以上类别",
}

# ============================================================
# 抽样策略：按书均匀分布，覆盖不同页码段
# ============================================================

def sample_pairs(merged_file: Path, n: int) -> list:
    """
    从三本书中按比例抽取 n 条，
    同时尽量覆盖每本书的前/中/后段页码。
    """
    by_book = {}
    with open(merged_file, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())
            book_id = record["book_id"]
            if book_id not in by_book:
                by_book[book_id] = []
            by_book[book_id].append(record)

    sampled = []
    books = list(by_book.keys())
    per_book = n // len(books)
    remainder = n % len(books)

    for i, book_id in enumerate(books):
        pairs = by_book[book_id]
        # 分三段：前1/3、中1/3、后1/3，各取 per_book/3 条
        seg_size = len(pairs) // 3
        quota = per_book + (1 if i < remainder else 0)
        per_seg = quota // 3
        extra = quota % 3

        selected = []
        for seg_idx in range(3):
            start = seg_idx * seg_size
            end = start + seg_size if seg_idx < 2 else len(pairs)
            seg = pairs[start:end]
            take = per_seg + (1 if seg_idx < extra else 0)
            selected.extend(random.sample(seg, min(take, len(seg))))

        # 补足到 quota（如果某段太小）
        if len(selected) < quota:
            remaining = [p for p in pairs if p not in selected]
            selected.extend(random.sample(remaining, min(quota - len(selected), len(remaining))))

        sampled.extend(selected[:quota])
        print(f"  {book_id}: 抽取 {len(selected[:quota])} 条（共 {len(pairs)} 条）")

    random.shuffle(sampled)
    return sampled


# ============================================================
# 图片处理
# ============================================================

def compress_image_to_base64(image_path: str, max_size: int = IMAGE_MAX_SIZE) -> str | None:
    """压缩图片并转为 base64（铁律2）"""
    try:
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"  ⚠️  图片处理失败 {image_path}: {e}")
        return None


# ============================================================
# 构建分类 Prompt
# ============================================================

def build_classification_prompt(caption: str, context: str) -> str:
    categories_text = "\n".join(
        f'  "{k}": {v}' for k, v in COARSE_CATEGORIES.items()
    )
    caption_part = f"图片说明文字：{caption}" if caption.strip() else "图片说明文字：（无）"
    context_part = f"周边正文文本：{context[:200]}" if context.strip() else "周边正文文本：（无）"

    return f"""你是中国传统插花教材图像分类专家。请根据图片内容和文字信息，从以下6个类别中选择最合适的一个。

类别定义：
{categories_text}

判断优先规则（按优先级排序，规则1最高）：
1. 【文字线索最优先】如果图注或正文含有「步骤」「步骤1/2/3」「步骤图」等字样，无论图片内容如何，优先选 diagram
2. 【文字线索次优先】如果图注含有「赏析」「作品」「速写」「手绘」等字样，优先选 arrangement
3. 【视觉判断】只要画面主体是已完成的插花作品（花材插在花器中形成完整作品），选 arrangement
4. 【vessel 严格限定】vessel 仅用于「空花器单独展示」，图中必须几乎没有花材；操作步骤中的花器图不算 vessel
5. 【diagram 宽松认定】步骤图即使没有箭头，只要图注有步骤序号，就归 diagram
6. 【other 最后手段】other 仅在确实无法归类时使用

参考信息（可能为空，以图片为准）：
{caption_part}
{context_part}

请只输出以下 JSON 格式，不要任何额外文字：
{{"label": "类别名", "confidence": 0.0到1.0的数字, "reason": "一句话理由"}}"""


# ============================================================
# 调用 Ollama
# ============================================================

def call_ollama(image_b64: str, prompt: str) -> dict:
    """调用本地 Ollama VLM，返回解析后的结果"""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "options": {"temperature": 0.1},  # 低温度，结果更稳定
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
        resp.raise_for_status()
        raw_text = resp.json().get("response", "").strip()

        # 解析 JSON 输出
        # 容错：模型有时会在 JSON 前后加多余文字
        start = raw_text.find("{")
        end = raw_text.rfind("}") + 1
        if start >= 0 and end > start:
            result = json.loads(raw_text[start:end])
            # 验证 label 是否合法
            if result.get("label") not in COARSE_CATEGORIES:
                result["label"] = "other"
                result["parse_warning"] = f"invalid label, original: {raw_text}"
            return result
        else:
            return {"label": "other", "confidence": 0.0, "reason": "parse_failed", "raw": raw_text}

    except requests.exceptions.Timeout:
        return {"label": "TIMEOUT", "confidence": 0.0, "reason": "request timeout"}
    except Exception as e:
        return {"label": "ERROR", "confidence": 0.0, "reason": str(e)}


# ============================================================
# 断点续传：读取已完成的 pair_id
# ============================================================

def load_done_ids(output_file: Path) -> set:
    done = set()
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    done.add(record["global_pair_id"])
                except:
                    pass
    return done


# ============================================================
# 主流程
# ============================================================

def main():
    random.seed(42)  # 固定随机种子，结果可复现
    SANDBOX_DIR.mkdir(parents=True, exist_ok=True)

    print(f"🌸 CTFAD Sandbox Classification")
    print(f"   模型：{MODEL_NAME}")
    print(f"   抽样数量：{SAMPLE_SIZE}")
    print(f"   输出：{OUTPUT_FILE}")
    print()

    # 抽样
    print("📋 抽样中...")
    samples = sample_pairs(MERGED_FILE, SAMPLE_SIZE)
    print(f"   共抽取 {len(samples)} 条\n")

    # 断点续传
    done_ids = load_done_ids(OUTPUT_FILE)
    if done_ids:
        print(f"⚡ 断点续传：已完成 {len(done_ids)} 条，跳过\n")

    # 分类
    success, failed, skipped = 0, 0, 0

    with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f:
        for i, pair in enumerate(samples):
            gid = pair["global_pair_id"]

            # 断点续传跳过
            if gid in done_ids:
                skipped += 1
                continue

            img_path = pair["image"]["absolute_path"]
            caption = pair["caption"].get("text", "") or ""
            context = pair["context"].get("text", "") or ""

            print(f"[{i+1:02d}/{len(samples)}] {gid} | {Path(img_path).name}")

            # 压缩图片
            img_b64 = compress_image_to_base64(img_path)
            if img_b64 is None:
                result = {"label": "ERROR", "confidence": 0.0, "reason": "image_load_failed"}
                failed += 1
            else:
                # 调用模型
                prompt = build_classification_prompt(caption, context)
                result = call_ollama(img_b64, prompt)

                if result["label"] in ("TIMEOUT", "ERROR"):
                    failed += 1
                    print(f"   ❌ {result['label']}: {result['reason']}")
                else:
                    success += 1
                    print(f"   → {result['label']} (置信度: {result.get('confidence', '?')})")

            # 写入结果（铁律3：处理一条写一条）
            output_record = {
                "global_pair_id": gid,
                "book_id": pair["book_id"],
                "image_path": img_path,
                "image_filename": Path(img_path).name,
                "caption": caption,
                "context": context[:300],
                "prediction": result,
                "predicted_at": datetime.now().isoformat(),
            }
            out_f.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            out_f.flush()

            # 铁律1：冷却
            time.sleep(SLEEP_BETWEEN)
            if (i + 1) % SLEEP_EVERY_N == 0:
                print(f"   💤 额外冷却 {SLEEP_EXTRA}s...")
                time.sleep(SLEEP_EXTRA)

    print(f"\n{'='*50}")
    print(f"✅ 沙盒分类完成")
    print(f"   成功：{success} | 失败：{failed} | 跳过：{skipped}")
    print(f"   结果文件：{OUTPUT_FILE}")
    print(f"   下一步：用 sandbox_review.html 审核结果")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
