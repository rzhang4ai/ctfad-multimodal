#!/usr/bin/env python3
"""
s2_classify_full.py
CTFAD Pipeline — Round 1 全量粗分类

功能：
  - 读取 merged_pairs.jsonl（432 条）
  - 自动跳过沙盒已处理的 40 条（断点续传）
  - 对剩余 392 条调用本地 qwen2.5vl:3b 进行粗分类
  - 输出 full_predictions.jsonl
  - 遵循三条铁律：sleep 冷却 + 断点续传 + 图片预压缩

用法：
  conda activate mineru
  python s2_classify_full.py

输出：
  /Volumes/aiworkbench/datasets/06_classified/full_predictions.jsonl
"""

import json
import time
import base64
import requests
from pathlib import Path
from datetime import datetime
from io import BytesIO
from PIL import Image

# ============================================================
# 配置
# ============================================================

DATASETS_ROOT = Path("/Volumes/aiworkbench/datasets")
MERGED_FILE   = DATASETS_ROOT / "06_classified" / "merged_pairs.jsonl"
SANDBOX_FILE  = DATASETS_ROOT / "06_classified" / "sandbox" / "sandbox_predictions.jsonl"
OUTPUT_FILE   = DATASETS_ROOT / "06_classified" / "full_predictions.jsonl"

OLLAMA_URL  = "http://localhost:11434/api/generate"
MODEL_NAME  = "qwen2.5vl:3b"

IMAGE_MAX_SIZE  = 768
SLEEP_BETWEEN   = 2.5
SLEEP_EVERY_N   = 20
SLEEP_EXTRA     = 10

# 优化后的 Round 1 粗分类类别定义（v2，含文字线索规则）
COARSE_CATEGORIES = {
    "arrangement": "插花作品图——画面主体是已完成的插花作品（含速写、手绘、古画等风格），无论实拍/手绘/古画",
    "vessel":      "花器图——画面主体是单独的空花器/容器，不含任何花材，也不是操作步骤",
    "plants":      "花材/植物图——画面主体是单独的花材、植物素材，没有花器",
    "photograph":  "其他实拍照片——场景、装饰物、文物等，不含插花作品",
    "diagram":     "示意图/步骤图——操作步骤图（即使无箭头）、构图比例图、结构说明图；图注含「步骤」「步骤N」字样的优先归此类",
    "other":       "无法归类——封面装饰、作者照、模糊图等，确实无法归入以上类别",
}

# ============================================================
# 工具函数
# ============================================================

def load_done_ids(output_file: Path, sandbox_file: Path) -> set:
    """断点续传：收集已完成的 global_pair_id（含沙盒结果）"""
    done = set()
    for f in [output_file, sandbox_file]:
        if f.exists():
            with open(f, "r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        done.add(json.loads(line.strip())["global_pair_id"])
                    except:
                        pass
    return done


def compress_image_to_base64(image_path: str, max_size: int = IMAGE_MAX_SIZE) -> str | None:
    try:
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"  ⚠️  图片处理失败：{e}")
        return None


def build_prompt(caption: str, context: str) -> str:
    categories_text = "\n".join(
        f'  "{k}": {v}' for k, v in COARSE_CATEGORIES.items()
    )
    caption_part = f"图片图注：{caption.strip()}" if caption.strip() else "图片图注：（无）"
    context_part = f"周边正文：{context.strip()[:200]}" if context.strip() else "周边正文：（无）"

    return f"""你是中国传统插花教材图像分类专家。请根据图片内容和文字信息，从以下6个类别中选择最合适的一个。

类别定义：
{categories_text}

判断优先规则（规则1优先级最高）：
1. 【文字线索最优先】图注或正文含「步骤」「步骤1/2/3」等字样 → 优先选 diagram
2. 【文字线索次优先】图注含「赏析」「作品」「速写」「手绘」等字样 → 优先选 arrangement
3. 【视觉判断】画面主体是已完成插花作品（花材插在花器中）→ 选 arrangement
4. 【vessel 严格限定】vessel 仅用于空花器单独展示，图中几乎没有花材
5. 【diagram 宽松认定】步骤图即使没有箭头，图注有步骤序号也归 diagram
6. 【other 最后手段】确实无法归类才选 other

参考文字信息（以图片为主，文字为辅）：
{caption_part}
{context_part}

请只输出以下 JSON 格式，不要任何额外文字：
{{"label": "类别名", "confidence": 0.0到1.0的数字, "reason": "一句话理由"}}"""


def call_ollama(image_b64: str, prompt: str) -> dict:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "options": {"temperature": 0.1},
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
        start, end = raw.find("{"), raw.rfind("}") + 1
        if start >= 0 and end > start:
            result = json.loads(raw[start:end])
            if result.get("label") not in COARSE_CATEGORIES:
                result["label"] = "other"
                result["parse_warning"] = f"invalid label: {raw}"
            return result
        return {"label": "other", "confidence": 0.0, "reason": "parse_failed", "raw": raw}
    except requests.exceptions.Timeout:
        return {"label": "TIMEOUT", "confidence": 0.0, "reason": "timeout"}
    except Exception as e:
        return {"label": "ERROR", "confidence": 0.0, "reason": str(e)}

# ============================================================
# 主流程
# ============================================================

def main():
    print(f"🌸 CTFAD Round 1 全量分类")
    print(f"   模型：{MODEL_NAME}")
    print(f"   输入：{MERGED_FILE}")
    print(f"   输出：{OUTPUT_FILE}\n")

    # 读取全量数据
    all_pairs = []
    with open(MERGED_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_pairs.append(json.loads(line))
    print(f"📋 全量数据：{len(all_pairs)} 条")

    # 断点续传：跳过沙盒已处理的 + 之前全量已处理的
    done_ids = load_done_ids(OUTPUT_FILE, SANDBOX_FILE)
    todo = [p for p in all_pairs if p["global_pair_id"] not in done_ids]
    print(f"⚡ 已完成（含沙盒）：{len(done_ids)} 条")
    print(f"   待处理：{len(todo)} 条\n")

    if not todo:
        print("✅ 所有数据已处理完成")
        return

    success, failed = 0, 0

    with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f:
        for i, pair in enumerate(todo):
            gid = pair["global_pair_id"]
            img_path = pair["image"]["absolute_path"]
            caption = pair["caption"].get("text", "") or ""
            context = pair["context"].get("text", "") or ""

            print(f"[{i+1:03d}/{len(todo)}] {gid} | {Path(img_path).name}")

            # 压缩图片
            img_b64 = compress_image_to_base64(img_path)
            if img_b64 is None:
                result = {"label": "ERROR", "confidence": 0.0, "reason": "image_load_failed"}
                failed += 1
            else:
                prompt = build_prompt(caption, context)
                result = call_ollama(img_b64, prompt)

                if result["label"] in ("TIMEOUT", "ERROR"):
                    failed += 1
                    print(f"   ❌ {result['label']}: {result['reason']}")
                else:
                    success += 1
                    print(f"   → {result['label']} ({result.get('confidence', '?')})")

            # 写入（铁律3：处理一条写一条）
            record = {
                "global_pair_id": gid,
                "book_id": pair["book_id"],
                "image_path": img_path,
                "image_filename": Path(img_path).name,
                "caption": caption,
                "context": context[:300],
                "prediction": result,
                "predicted_at": datetime.now().isoformat(),
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

            # 铁律1：冷却
            time.sleep(SLEEP_BETWEEN)
            if (i + 1) % SLEEP_EVERY_N == 0:
                print(f"   💤 额外冷却 {SLEEP_EXTRA}s...")
                time.sleep(SLEEP_EXTRA)

    print(f"\n{'='*50}")
    print(f"✅ 全量分类完成")
    print(f"   成功：{success} | 失败：{failed}")
    print(f"   输出：{OUTPUT_FILE}")
    print(f"   下一步：运行 s2_merge_results.py 合并沙盒标注")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
