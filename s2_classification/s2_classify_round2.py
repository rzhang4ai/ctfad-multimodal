#!/usr/bin/env python3
"""
s2_classify_round2.py
CTFAD Pipeline — Round 2 细分类（自动流水线）

功能：
  - 读取 Round 1 全量结果（full_predictions.jsonl）
  - 自动跳过沙盒已有人工标注的 40 条
  - 对需要细分的 5 类（arrangement/vessel/plants/photograph/diagram）跑细分类
  - other 类不细分，直接保留粗分类标签
  - 输出 round2_predictions.jsonl

细分类体系：
  arrangement → arrangement_photo / arrangement_drawing / arrangement_painting
  vessel      → vessel_photo / vessel_drawing / vessel_painting
  plants      → plants_photo / plants_drawing / plants_painting
  photograph  → decoration / scene_photo / historical_artifact
  diagram     → step_diagram / layout_diagram / structure_diagram
  other       → 保持 other 不细分

用法：
  conda activate mineru
  python s2_classify_round2.py
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

DATASETS_ROOT   = Path("/Volumes/aiworkbench/datasets")
FULL_PRED_FILE  = DATASETS_ROOT / "06_classified" / "full_predictions.jsonl"
MERGED_FILE     = DATASETS_ROOT / "06_classified" / "merged_pairs.jsonl"
SANDBOX_FILE    = DATASETS_ROOT / "06_classified" / "sandbox" / "sandbox_reviewed.jsonl"
OUTPUT_FILE     = DATASETS_ROOT / "06_classified" / "round2_predictions.jsonl"

OLLAMA_URL  = "http://localhost:11434/api/generate"
MODEL_NAME  = "qwen2.5vl:3b"

IMAGE_MAX_SIZE = 768
SLEEP_BETWEEN  = 2.5
SLEEP_EVERY_N  = 20
SLEEP_EXTRA    = 10

# ============================================================
# 细分类定义（每个粗分类对应的子类别 + prompt）
# ============================================================

FINE_CATEGORIES = {

    "arrangement": {
        "labels": {
            "arrangement_photo":    "实拍照片——完成的插花作品实物照片，有真实的花材质感和光影",
            "arrangement_drawing":  "手绘/线描/速写——手绘风格的插花作品图，线条感明显，图注常含「速写」「手绘」「作品赏析」",
            "arrangement_painting": "古画/传统绘画——中国传统绘画风格（工笔、写意、版画等），宋明清朝代风格",
        },
        "hint": "判断依据：实拍照片质感真实；手绘有明显线条/速写感；古画有传统绘画风格（墨色、宣纸感）。图注含「速写」「赏析」→ drawing。",
    },

    "vessel": {
        "labels": {
            "vessel_photo":    "实拍照片——花器实物照片，有真实质感和光影",
            "vessel_drawing":  "手绘/线稿——花器的手绘图或线稿，常用于说明器型",
            "vessel_painting": "传统绘画——古画中单独出现的花器图像",
        },
        "hint": "判断依据：与 arrangement 相同，区别实拍/手绘/古画风格即可。",
    },

    "plants": {
        "labels": {
            "plants_photo":    "实拍照片——花材/植物素材的实物照片",
            "plants_drawing":  "手绘/线稿——花材的手绘图，常见于花材说明页",
            "plants_painting": "传统绘画——古画中单独出现的植物图像（花卉谱、本草图谱等）",
        },
        "hint": "判断依据：区别实拍/手绘/古画风格。",
    },

    "photograph": {
        "labels": {
            "decoration":          "装饰物/配饰/案几/陈设——茶道配件、香炉、山石、案几等，不含花器和插花",
            "scene_photo":         "场景/环境照片——展厅全景、茶席整体、室内空间，插花（若有）不是画面绝对主体",
            "historical_artifact": "历史文物/考古实物——博物馆馆藏、出土文物，有明显文物属性（标签、考古背景等）",
        },
        "hint": "判断依据：画面主体是器物特写→decoration；画面是整体空间环境→scene_photo；有明显博物馆/考古背景→historical_artifact。",
    },

    "diagram": {
        "labels": {
            "step_diagram":      "操作步骤图——展示插花操作步骤、技法演示，含序号或步骤说明，图注常含「步骤N」",
            "layout_diagram":    "构图/比例示意图——展示角度、比例、空间关系，含角度线、黄金分割线等",
            "structure_diagram": "结构/解剖示意图——花器内部结构、花材解剖、固定原理等技术性图解",
        },
        "hint": "判断依据：图注含「步骤」→step；含角度线/比例线→layout；展示内部结构或解剖→structure。",
    },
}

# 不需要细分的类别
NO_FINE_SPLIT = {"other"}


# ============================================================
# 工具函数
# ============================================================

def load_sandbox_human_labels(sandbox_file: Path) -> dict:
    """读取沙盒人工标注：{global_pair_id: human_fine_label}"""
    labels = {}
    if not sandbox_file.exists():
        return labels
    with open(sandbox_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            gid = r.get("global_pair_id")
            human_label = r.get("human_label")
            if gid and human_label:
                labels[gid] = human_label
    print(f"   沙盒人工标注：{len(labels)} 条（将直接复用）")
    return labels


def load_merged_pairs(merged_file: Path) -> dict:
    """读取原始合并数据：{global_pair_id: pair}，用于获取完整图文信息"""
    pairs = {}
    with open(merged_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                p = json.loads(line)
                pairs[p["global_pair_id"]] = p
    return pairs


def load_done_ids(output_file: Path) -> set:
    done = set()
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line.strip())["global_pair_id"])
                except:
                    pass
    return done


def compress_image_to_base64(image_path: str, max_size: int = IMAGE_MAX_SIZE):
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


def build_fine_prompt(coarse_label: str, caption: str, context: str) -> str:
    spec = FINE_CATEGORIES[coarse_label]
    categories_text = "\n".join(
        f'  "{k}": {v}' for k, v in spec["labels"].items()
    )
    hint = spec["hint"]
    caption_part = f"图注：{caption.strip()}" if caption.strip() else "图注：（无）"
    context_part = f"正文：{context.strip()[:200]}" if context.strip() else "正文：（无）"

    return f"""你是中国传统插花教材图像分类专家。
这张图已确认属于「{coarse_label}」大类，请进一步从以下子类别中选择最合适的一个。

子类别定义：
{categories_text}

判断提示：{hint}

参考文字信息（以图片为主，文字为辅）：
{caption_part}
{context_part}

请只输出以下 JSON 格式，不要任何额外文字：
{{"label": "子类别名", "confidence": 0.0到1.0的数字, "reason": "一句话理由"}}"""


def call_ollama(image_b64: str, prompt: str, valid_labels: set) -> dict:
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
            if result.get("label") not in valid_labels:
                # 取第一个有效标签作为兜底
                result["label"] = next(iter(valid_labels))
                result["parse_warning"] = f"invalid label: {raw}"
            return result
        return {"label": next(iter(valid_labels)), "confidence": 0.0,
                "reason": "parse_failed", "raw": raw}
    except requests.exceptions.Timeout:
        return {"label": "TIMEOUT", "confidence": 0.0, "reason": "timeout"}
    except Exception as e:
        return {"label": "ERROR", "confidence": 0.0, "reason": str(e)}


# ============================================================
# 主流程
# ============================================================

def main():
    print(f"🌸 CTFAD Round 2 细分类")
    print(f"   模型：{MODEL_NAME}\n")

    # 读取沙盒人工标注
    sandbox_labels = load_sandbox_human_labels(SANDBOX_FILE)

    # 读取原始图文数据（获取 caption/context）
    merged_pairs = load_merged_pairs(MERGED_FILE)

    # 断点续传
    done_ids = load_done_ids(OUTPUT_FILE)

    # 读取 Round 1 结果
    round1_records = []
    with open(FULL_PRED_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                round1_records.append(json.loads(line))

    print(f"📋 Round 1 结果：{len(round1_records)} 条")
    print(f"   已完成（断点续传）：{len(done_ids)} 条")

    # 统计各粗分类数量
    coarse_dist = {}
    for r in round1_records:
        label = r.get("prediction", {}).get("label", "other")
        coarse_dist[label] = coarse_dist.get(label, 0) + 1
    print(f"\n   Round 1 粗分类分布：")
    for label, count in sorted(coarse_dist.items(), key=lambda x: -x[1]):
        needs_fine = "→ 细分" if label not in NO_FINE_SPLIT else "→ 不细分"
        print(f"   {label}: {count} 条 {needs_fine}")

    success, skipped_sandbox, skipped_no_split, failed = 0, 0, 0, 0

    with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f:
        for i, r1 in enumerate(round1_records):
            gid = r1["global_pair_id"]
            coarse_label = r1.get("prediction", {}).get("label", "other")
            coarse_conf  = r1.get("prediction", {}).get("confidence", 0.0)

            # 断点续传
            if gid in done_ids:
                continue

            # ── 情况1：沙盒已有人工细分类标注 ──
            if gid in sandbox_labels:
                fine_label = sandbox_labels[gid]
                record = {
                    "global_pair_id":   gid,
                    "book_id":          r1["book_id"],
                    "image_path":       r1["image_path"],
                    "image_filename":   r1["image_filename"],
                    "caption":          r1.get("caption", ""),
                    "context":          r1.get("context", ""),
                    "coarse_label":     coarse_label,
                    "coarse_confidence": coarse_conf,
                    "fine_label":       fine_label,
                    "fine_confidence":  1.0,
                    "fine_reason":      "human_sandbox_annotation",
                    "label_source":     "human_sandbox",
                    "predicted_at":     datetime.now().isoformat(),
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                skipped_sandbox += 1
                continue

            # ── 情况2：other 类不细分 ──
            if coarse_label in NO_FINE_SPLIT:
                record = {
                    "global_pair_id":    gid,
                    "book_id":           r1["book_id"],
                    "image_path":        r1["image_path"],
                    "image_filename":    r1["image_filename"],
                    "caption":           r1.get("caption", ""),
                    "context":           r1.get("context", ""),
                    "coarse_label":      coarse_label,
                    "coarse_confidence": coarse_conf,
                    "fine_label":        coarse_label,  # 细分类 = 粗分类
                    "fine_confidence":   coarse_conf,
                    "fine_reason":       "no_fine_split_for_this_category",
                    "label_source":      "model_round1_only",
                    "predicted_at":      datetime.now().isoformat(),
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                skipped_no_split += 1
                continue

            # ── 情况3：需要细分类 ──
            img_path = r1["image_path"]
            # 从 merged_pairs 获取最新的 caption/context
            pair = merged_pairs.get(gid, {})
            caption = pair.get("caption", {}).get("text", "") or r1.get("caption", "") or ""
            context = pair.get("context", {}).get("text", "") or r1.get("context", "") or ""

            print(f"[{i+1:03d}] {gid} | {coarse_label} → ?")

            img_b64 = compress_image_to_base64(img_path)
            if img_b64 is None:
                result = {"label": "ERROR", "confidence": 0.0, "reason": "image_load_failed"}
                failed += 1
            else:
                valid_labels = set(FINE_CATEGORIES[coarse_label]["labels"].keys())
                prompt = build_fine_prompt(coarse_label, caption, context)
                result = call_ollama(img_b64, prompt, valid_labels)

                if result["label"] in ("TIMEOUT", "ERROR"):
                    failed += 1
                    print(f"   ❌ {result['label']}")
                else:
                    success += 1
                    print(f"   → {result['label']} ({result.get('confidence', '?')})")

            record = {
                "global_pair_id":    gid,
                "book_id":           r1["book_id"],
                "image_path":        img_path,
                "image_filename":    r1["image_filename"],
                "caption":           caption,
                "context":           context[:300],
                "coarse_label":      coarse_label,
                "coarse_confidence": coarse_conf,
                "fine_label":        result["label"],
                "fine_confidence":   result.get("confidence", 0.0),
                "fine_reason":       result.get("reason", ""),
                "label_source":      "model_round2",
                "predicted_at":      datetime.now().isoformat(),
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

            time.sleep(SLEEP_BETWEEN)
            if (success + failed) % SLEEP_EVERY_N == 0 and (success + failed) > 0:
                print(f"   💤 额外冷却 {SLEEP_EXTRA}s...")
                time.sleep(SLEEP_EXTRA)

    print(f"\n{'='*50}")
    print(f"✅ Round 2 细分类完成")
    print(f"   模型细分：{success} 条")
    print(f"   沙盒复用：{skipped_sandbox} 条")
    print(f"   不细分类：{skipped_no_split} 条")
    print(f"   失败：    {failed} 条")
    print(f"   输出：{OUTPUT_FILE}")
    print(f"   下一步：运行 s2_merge_results.py")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
