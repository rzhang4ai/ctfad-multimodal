#!/usr/bin/env python3
"""
s3_assess_image_quality.py
CTFAD S3 v1 — Golden Test Set 图像质量评估

用途：对 Golden Test Set 图片进行全面质量分析，
      产出一份 HTML 报告 + quality_report.json，
      为后续 3D 生成实验提供数据基础对比。

运行：
    conda activate ctfad-s3-exp  (或 ctfad-s3)
    python s3_assess_image_quality.py

输出：
    datasets/07_golden_test_set/reports/quality_report.json  ← 机器可读
    datasets/07_golden_test_set/reports/quality_report.html  ← 可视化报告

v0 原始版本保留在 s3_experiments_v0/s3_assess_image_quality.py
"""

import os
import json
import math
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────
# 路径配置（按需修改）
# ─────────────────────────────────────────────
BASE_DIR = Path("/Volumes/aiworkbench/datasets/07_golden_test_set")
INPUT_DIR  = BASE_DIR / "images"          # 输入：Golden Test Set 图片
OUTPUT_DIR = BASE_DIR / "reports"         # 输出：报告

# ─────────────────────────────────────────────
# 图片来源分组（用于报告中对比显示）
# 格式：{ "文件名前缀或关键字": "来源标签" }
# 如不需要分组，留空 dict 即可，会自动归入 "unknown"
# ─────────────────────────────────────────────
SOURCE_MAPPING = {
    # 书籍扫描子集（从 arrangement_photo 挑出的）
    "img_p": "书籍扫描",
    # 个人学习照片（手动命名）
    "ZHHY": "个人学习照",
    "my_":      "个人学习照",
    # 中华花艺文家基金会官网图
    "cfaf":     "CFAF官网",
    "foundation": "CFAF官网",
}

# ─────────────────────────────────────────────
# 依赖检查
# ─────────────────────────────────────────────
try:
    import cv2
    import numpy as np
    from PIL import Image
except ImportError:
    print("❌ 缺少依赖，请运行：")
    print("   pip install opencv-python-headless pillow numpy --break-system-packages")
    exit(1)


def get_source_label(filename: str) -> str:
    """根据文件名判断来源分组"""
    name_lower = filename.lower()
    for key, label in SOURCE_MAPPING.items():
        if key.lower() in name_lower:
            return label
    return "其他"


def assess_single_image(img_path: Path) -> dict:
    """对单张图片进行全面质量评估，返回指标字典"""
    result = {
        "filename": img_path.name,
        "filepath": str(img_path),
        "source": get_source_label(img_path.name),
        "exists": False,
        "error": None,
    }

    # ── 基础文件信息 ──────────────────────────
    if not img_path.exists():
        result["error"] = "文件不存在"
        return result

    result["exists"] = True
    result["file_size_kb"] = round(img_path.stat().st_size / 1024, 1)

    # ── PIL 基础信息 ──────────────────────────
    try:
        pil_img = Image.open(img_path)
        result["width"]  = pil_img.width
        result["height"] = pil_img.height
        result["mode"]   = pil_img.mode          # RGB / RGBA / L / etc.
        result["format"] = pil_img.format or img_path.suffix.upper().lstrip(".")
        result["megapixels"] = round(pil_img.width * pil_img.height / 1_000_000, 2)
        result["aspect_ratio"] = round(pil_img.width / pil_img.height, 3)
        pil_img.close()
    except Exception as e:
        result["error"] = f"PIL 读取失败: {e}"
        return result

    # ── OpenCV 深度分析 ──────────────────────
    try:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            result["error"] = "OpenCV 无法读取（可能是 HEIC/WebP 等格式）"
            return result

        # 1. 模糊度（拉普拉斯算子方差，越高越清晰）
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        result["blur_variance"] = round(blur_var, 1)
        result["is_blurry"] = blur_var < 100

        # 2. 灰度检测（S 通道均值）
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mean_saturation = float(hsv[:, :, 1].mean())
        result["mean_saturation"] = round(mean_saturation, 1)
        result["is_grayscale"] = mean_saturation < 10

        # 3. 亮度（V 通道均值）
        mean_value = float(hsv[:, :, 2].mean())
        result["mean_brightness"] = round(mean_value, 1)
        result["is_too_dark"]  = mean_value < 50
        result["is_too_bright"] = mean_value > 230

        # 4. 对比度（灰度图标准差）
        contrast = float(gray.std())
        result["contrast_std"] = round(contrast, 1)
        result["is_low_contrast"] = contrast < 30

        # 5. 背景复杂度评估
        # 取图像四角 10% 区域，计算颜色方差
        h, w = img_bgr.shape[:2]
        margin_h = max(1, h // 10)
        margin_w = max(1, w // 10)
        corners = [
            img_bgr[:margin_h, :margin_w],
            img_bgr[:margin_h, w-margin_w:],
            img_bgr[h-margin_h:, :margin_w],
            img_bgr[h-margin_h:, w-margin_w:],
        ]
        corner_pixels = np.concatenate([c.reshape(-1, 3) for c in corners], axis=0)
        bg_color_std = float(corner_pixels.std())
        result["bg_complexity"] = round(bg_color_std, 1)
        # 背景越简单（低方差），越适合去背景
        result["bg_simple"] = bg_color_std < 30

        # 6. 主体占比估算（中心区域 vs 边缘亮度差）
        center_region = img_bgr[h//4:3*h//4, w//4:3*w//4]
        center_brightness = float(cv2.cvtColor(center_region, cv2.COLOR_BGR2HSV)[:,:,2].mean())
        edge_brightness = mean_value
        result["center_vs_edge_brightness"] = round(center_brightness - edge_brightness, 1)

        # 7. 噪点评估（高频成分占比）
        # 用高斯模糊前后差异来估算
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise_diff = float(cv2.absdiff(gray, blurred).mean())
        result["noise_level"] = round(noise_diff, 2)
        result["is_noisy"] = noise_diff > 8

        # 8. 综合质量评级
        result["quality_tier"], result["quality_issues"] = compute_quality_tier(result)

    except Exception as e:
        result["error"] = f"OpenCV 分析失败: {e}"

    return result


def compute_quality_tier(r: dict) -> tuple:
    """根据指标计算综合质量等级和问题列表"""
    issues = []

    if r.get("is_blurry"):
        issues.append(f"模糊（blur={r.get('blur_variance', '?')}）")
    if r.get("is_grayscale"):
        issues.append("灰度图/黑白")
    if r.get("is_too_dark"):
        issues.append(f"过暗（brightness={r.get('mean_brightness', '?')}）")
    if r.get("is_too_bright"):
        issues.append(f"过曝（brightness={r.get('mean_brightness', '?')}）")
    if r.get("is_low_contrast"):
        issues.append(f"低对比度（contrast={r.get('contrast_std', '?')}）")
    if r.get("is_noisy"):
        issues.append(f"噪点较多（noise={r.get('noise_level', '?')}）")

    # 分辨率检查
    w = r.get("width", 0)
    h = r.get("height", 0)
    short_edge = min(w, h)
    if short_edge < 512:
        issues.append(f"分辨率过低（{w}×{h}，短边<512px）")
    elif short_edge < 768:
        issues.append(f"分辨率偏低（{w}×{h}，短边<768px）")

    # 评级
    if len(issues) == 0:
        tier = "A"
    elif len(issues) == 1 and "模糊" not in issues[0] and "灰度" not in issues[0]:
        tier = "A-"  # 轻微问题
    elif len(issues) <= 2 and "灰度" not in str(issues):
        tier = "B"
    else:
        tier = "C"

    return tier, issues


def format_score_bar(value: float, max_val: float, color: str = "#8fba5c") -> str:
    """生成 HTML 进度条"""
    pct = min(100, (value / max_val) * 100)
    return f'<div style="background:#2a2b27;border-radius:2px;height:6px;width:100%;margin-top:4px"><div style="background:{color};width:{pct:.0f}%;height:100%;border-radius:2px"></div></div>'


def generate_html_report(results: list, output_path: Path):
    """生成可视化 HTML 报告"""

    # 统计摘要
    total = len(results)
    valid = [r for r in results if r.get("exists") and not r.get("error")]
    tier_counts = {}
    for r in valid:
        t = r.get("quality_tier", "?")
        tier_counts[t] = tier_counts.get(t, 0) + 1

    # 排序：按综合评级 + 模糊度
    valid_sorted = sorted(valid, key=lambda r: (
        {"A": 0, "A-": 1, "B": 2, "C": 3}.get(r.get("quality_tier", "C"), 3),
        -r.get("blur_variance", 0)
    ))

    cards_html = ""
    for r in valid_sorted:
        tier = r.get("quality_tier", "?")
        tier_color = {"A": "#8fba5c", "A-": "#a8d070", "B": "#d4a853", "C": "#c4534a"}.get(tier, "#888")
        issues = r.get("quality_issues", [])
        issues_html = "".join(f'<span style="background:#c4534a22;color:#c4534a;border:1px solid #c4534a44;padding:2px 7px;border-radius:2px;font-size:10px;margin:2px 2px 0 0;display:inline-block">{i}</span>' for i in issues) if issues else '<span style="color:#8fba5c;font-size:11px">✅ 无问题</span>'

        blur = r.get("blur_variance", 0)
        sat  = r.get("mean_saturation", 0)
        brt  = r.get("mean_brightness", 0)
        ctr  = r.get("contrast_std", 0)
        bg   = r.get("bg_complexity", 0)
        noise = r.get("noise_level", 0)

        # 背景复杂度颜色（越低越好）
        bg_color = "#8fba5c" if bg < 30 else "#d4a853" if bg < 60 else "#c4534a"
        bg_label = "简单✅" if bg < 30 else "中等⚠️" if bg < 60 else "复杂❌"

        cards_html += f"""
        <div style="background:#1c1d1a;border:1px solid #2a2b27;border-radius:4px;overflow:hidden;border-top:3px solid {tier_color}">
          <div style="padding:12px 14px;border-bottom:1px solid #2a2b27">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:6px">
              <div>
                <div style="font-size:12px;color:#d4d2c8;word-break:break-all">{r['filename']}</div>
                <div style="font-size:10px;color:#6b6a62;margin-top:2px">{r.get('source','?')} · {r.get('width','?')}×{r.get('height','?')}px · {r.get('file_size_kb','?')}KB · {r.get('megapixels','?')}MP</div>
              </div>
              <div style="background:{tier_color}22;color:{tier_color};border:1px solid {tier_color};padding:3px 10px;border-radius:2px;font-size:13px;font-weight:bold;white-space:nowrap;margin-left:12px">
                {tier} 级
              </div>
            </div>
            <div style="margin-top:6px">{issues_html}</div>
          </div>
          <div style="padding:12px 14px">
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">

              <div>
                <div style="display:flex;justify-content:space-between;font-size:10px;color:#6b6a62"><span>模糊度</span><span style="color:#d4d2c8">{blur:.0f}</span></div>
                {format_score_bar(min(blur,700), 700, "#8fba5c" if blur>=100 else "#c4534a")}
              </div>

              <div>
                <div style="display:flex;justify-content:space-between;font-size:10px;color:#6b6a62"><span>色彩饱和度</span><span style="color:#d4d2c8">{sat:.0f}</span></div>
                {format_score_bar(sat, 255, "#8fba5c" if sat>=10 else "#c4534a")}
              </div>

              <div>
                <div style="display:flex;justify-content:space-between;font-size:10px;color:#6b6a62"><span>亮度</span><span style="color:#d4d2c8">{brt:.0f}</span></div>
                {format_score_bar(brt, 255, "#8fba5c" if 50<=brt<=230 else "#d4a853")}
              </div>

              <div>
                <div style="display:flex;justify-content:space-between;font-size:10px;color:#6b6a62"><span>对比度</span><span style="color:#d4d2c8">{ctr:.0f}</span></div>
                {format_score_bar(ctr, 100, "#8fba5c" if ctr>=30 else "#d4a853")}
              </div>

              <div>
                <div style="display:flex;justify-content:space-between;font-size:10px;color:#6b6a62"><span>背景复杂度</span><span style="color:{bg_color}">{bg:.0f} {bg_label}</span></div>
                {format_score_bar(min(bg,100), 100, bg_color)}
              </div>

              <div>
                <div style="display:flex;justify-content:space-between;font-size:10px;color:#6b6a62"><span>噪点</span><span style="color:#d4d2c8">{noise:.1f}</span></div>
                {format_score_bar(min(noise,15), 15, "#8fba5c" if noise<=8 else "#d4a853")}
              </div>

            </div>

            <div style="margin-top:10px;padding-top:8px;border-top:1px solid #2a2b27;font-size:10px;color:#6b6a62">
              尺寸比例 {r.get('aspect_ratio','?')} ·
              {'<span style="color:#8fba5c">背景简单，去背景友好</span>' if r.get('bg_simple') else '<span style="color:#d4a853">背景较复杂</span>'} ·
              {'<span style="color:#8fba5c">彩色</span>' if not r.get('is_grayscale') else '<span style="color:#c4534a">灰度</span>'}
            </div>
          </div>
        </div>
        """

    # 无效文件
    invalid = [r for r in results if not r.get("exists") or r.get("error")]
    invalid_html = ""
    for r in invalid:
        invalid_html += f'<div style="padding:8px 12px;background:#c4534a11;border:1px solid #c4534a33;border-radius:3px;margin-bottom:6px;font-size:11px;color:#c4534a">{r["filename"]} — {r.get("error","未知错误")}</div>'

    _tier_colors = {"A": "#8fba5c", "A-": "#a8d070", "B": "#d4a853", "C": "#c4534a"}
    tier_summary = " · ".join(f'<span style="color:{_tier_colors.get(t,"#888")}">{t}级: {n}张</span>' for t, n in sorted(tier_counts.items()))

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>CTFAD S3 图像质量报告</title>
<style>
  body {{ background:#0e0f0d; color:#d4d2c8; font-family:'SF Mono','Monaco',monospace; font-size:13px; margin:0; padding:0 }}
  .header {{ background:#161714; border-bottom:1px solid #2a2b27; padding:20px 32px; }}
  .title {{ font-size:18px; color:#8fba5c; font-family:'Hiragino Serif','STSong',serif; margin-bottom:4px }}
  .subtitle {{ font-size:11px; color:#6b6a62; letter-spacing:0.08em }}
  .summary {{ background:#1c1d1a; border:1px solid #2a2b27; border-radius:4px; padding:14px 18px; margin:24px 32px 16px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr)); gap:16px; padding:0 32px 32px }}
  ::-webkit-scrollbar {{ width:5px }} ::-webkit-scrollbar-track {{ background:transparent }} ::-webkit-scrollbar-thumb {{ background:#2a2b27; border-radius:2px }}
</style>
</head>
<body>
<div class="header">
  <div class="title">CTFAD · S3 测试图像质量报告</div>
  <div class="subtitle">生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')} · 总计 {total} 张 · 有效 {len(valid)} 张</div>
</div>

<div class="summary">
  <div style="font-size:11px;color:#6b6a62;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px">质量分布摘要</div>
  <div style="font-size:13px">{tier_summary}</div>
  <div style="margin-top:10px;font-size:11px;color:#6b6a62">
    💡 <strong style="color:#d4d2c8">背景复杂度</strong>是去背景成功率的关键指标。
    A 级 + 背景简单 的图片是 3D 生成实验的最优输入。
  </div>
</div>

{'<div style="padding:0 32px 16px"><div style="font-size:11px;color:#6b6a62;margin-bottom:8px">⚠️ 读取失败</div>' + invalid_html + '</div>' if invalid_html else ''}

<div class="grid">
  {cards_html}
</div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    print("=" * 55)
    print("CTFAD S3 — 测试图像质量评估")
    print("=" * 55)

    # 找到所有图片
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
    img_files = sorted([
        p for p in INPUT_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    ])

    if not img_files:
        print(f"❌ 在 {INPUT_DIR} 未找到图片文件")
        print("   支持格式：JPG / PNG / WebP / BMP / TIFF")
        return

    print(f"📂 输入目录：{INPUT_DIR}")
    print(f"🖼  找到 {len(img_files)} 张图片")
    print()

    results = []
    for i, img_path in enumerate(img_files, 1):
        print(f"  [{i:2d}/{len(img_files)}] 分析 {img_path.name}...", end="", flush=True)
        r = assess_single_image(img_path)
        results.append(r)

        if r.get("error"):
            print(f" ❌ {r['error']}")
        else:
            tier = r.get("quality_tier", "?")
            blur = r.get("blur_variance", 0)
            issues = r.get("quality_issues", [])
            bg_note = "背景简单" if r.get("bg_simple") else "背景复杂"
            print(f" {tier}级 | blur={blur:.0f} | {bg_note}" + (f" | ⚠️ {', '.join(issues)}" if issues else " | ✅"))

    # 保存 JSON
    json_path = OUTPUT_DIR / "quality_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 生成 HTML 报告
    html_path = OUTPUT_DIR / "quality_report.html"
    generate_html_report(results, html_path)

    # 控制台摘要
    print()
    print("=" * 55)
    valid = [r for r in results if r.get("exists") and not r.get("error")]
    tier_counts = {}
    for r in valid:
        t = r.get("quality_tier", "?")
        tier_counts[t] = tier_counts.get(t, 0) + 1

    print("📊 质量分布：")
    for tier in ["A", "A-", "B", "C"]:
        n = tier_counts.get(tier, 0)
        bar = "█" * n
        print(f"   {tier} 级：{n:2d} 张  {bar}")

    best = [r for r in valid if r.get("quality_tier") in ("A", "A-")]
    print(f"\n✅ 推荐用于 3D 实验的图（A/A- 级）：{len(best)} 张")
    for r in best:
        bg = "✅背景简单" if r.get("bg_simple") else "⚠️背景复杂"
        print(f"   · {r['filename']} ({r['width']}×{r['height']}px, blur={r.get('blur_variance',0):.0f}, {bg})")

    print(f"\n📄 JSON 报告：{json_path}")
    print(f"🌐 HTML 报告：{html_path}")
    print("   （用 Safari 打开 HTML 报告查看可视化结果）")
    print("=" * 55)


if __name__ == "__main__":
    main()
