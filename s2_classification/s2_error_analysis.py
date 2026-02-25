#!/usr/bin/env python3
"""
s2_error_analysis.py
CTFAD Pipeline — 错误案例分析

功能：
  - 读取 sandbox_reviewed.jsonl
  - 提取所有错误案例
  - 生成可视化 HTML 报告（含图片、预测标签、正确标签、文本信息）
  - 帮助判断是 prompt 问题还是模型能力问题

用法：
  conda activate mineru
  python s2_error_analysis.py

输出：
  /Volumes/aiworkbench/datasets/06_classified/sandbox/error_analysis.html
"""

import json
from pathlib import Path
from collections import defaultdict

SANDBOX_DIR = Path("/Volumes/aiworkbench/datasets/06_classified/sandbox")
REVIEWED_FILE = SANDBOX_DIR / "sandbox_reviewed.jsonl"
OUTPUT_HTML = SANDBOX_DIR / "error_analysis.html"

def main():
    if not REVIEWED_FILE.exists():
        print(f"❌ 找不到：{REVIEWED_FILE}")
        return

    # 读取所有记录
    records = []
    with open(REVIEWED_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    errors = [r for r in records if r.get("is_correct") == False]
    correct = [r for r in records if r.get("is_correct") == True]

    print(f"📊 总计：{len(records)} 条")
    print(f"   正确：{len(correct)} 条")
    print(f"   错误：{len(errors)} 条")

    # 按混淆对分组
    by_confusion = defaultdict(list)
    for r in errors:
        pred = r.get("predicted_label", "?")
        human = r.get("human_label", "?")
        by_confusion[(pred, human)].append(r)

    print(f"\n   混淆对分布：")
    for (pred, human), items in sorted(by_confusion.items(), key=lambda x: -len(x[1])):
        print(f"   {pred} → 应为 {human}：{len(items)} 次")

    # 生成 HTML
    html = generate_html(errors, by_confusion, len(records), len(correct))
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n✅ 错误分析报告已生成：")
    print(f"   {OUTPUT_HTML}")
    print(f"   用浏览器（Safari）打开查看")


def label_color(label):
    colors = {
        "arrangement": "#e94560",
        "vessel":       "#8b5cf6",
        "plants":       "#22c55e",
        "photograph":   "#3b82f6",
        "diagram":      "#f59e0b",
        "other":        "#9ca3af",
    }
    # 细分类也取粗分类颜色
    for key, color in colors.items():
        if label and label.startswith(key):
            return color
    return "#6b7280"


def generate_html(errors, by_confusion, total, correct_count):
    accuracy = correct_count / total * 100 if total > 0 else 0

    # 生成混淆对导航
    confusion_nav = ""
    for (pred, human), items in sorted(by_confusion.items(), key=lambda x: -len(x[1])):
        anchor = f"{pred}_to_{human}"
        confusion_nav += f"""
        <a href="#{anchor}" class="confusion-pill">
          <span class="pill-pred" style="background:{label_color(pred)}">{pred}</span>
          <span class="pill-arrow">→</span>
          <span class="pill-human" style="background:{label_color(human)}">{human}</span>
          <span class="pill-count">{len(items)}</span>
        </a>"""

    # 生成每个混淆组的卡片
    sections_html = ""
    for (pred, human), items in sorted(by_confusion.items(), key=lambda x: -len(x[1])):
        anchor = f"{pred}_to_{human}"
        cards_html = ""
        for r in items:
            img_path = r.get("image_path", "")
            img_src = f"file://{img_path}"
            gid = r.get("global_pair_id", "")
            book_id = r.get("book_id", "")
            caption = r.get("caption", "") or ""
            context = r.get("context", "") or ""
            confidence = r.get("predicted_confidence")
            conf_str = f"{confidence*100:.0f}%" if confidence is not None else "?"
            reason = r.get("predicted_reason", "") or ""

            text_snippet = ""
            if caption.strip():
                text_snippet += f"<div class='text-row'><span class='text-label'>图注</span>{caption[:150]}</div>"
            if context.strip():
                text_snippet += f"<div class='text-row'><span class='text-label'>正文</span>{context[:200]}</div>"
            if not text_snippet:
                text_snippet = "<div class='text-row empty'>（无文本信息）</div>"

            cards_html += f"""
        <div class="error-card">
          <img class="card-img"
               src="{img_src}"
               alt="{gid}"
               onclick="openLightbox('{img_src}')"
               onerror="this.style.background='#f3f4f6';this.style.height='160px';this.alt='图片无法加载'">
          <div class="card-body">
            <div class="card-meta">{gid} · {book_id}</div>
            <div class="label-row">
              <div class="label-block wrong-block">
                <div class="label-title">模型预测</div>
                <span class="label-badge" style="background:{label_color(pred)}">{pred}</span>
                <span class="conf">置信度 {conf_str}</span>
              </div>
              <div class="arrow-block">✕→</div>
              <div class="label-block correct-block">
                <div class="label-title">正确类别</div>
                <span class="label-badge" style="background:{label_color(human)}">{human}</span>
              </div>
            </div>
            {f'<div class="reason-row">模型理由：{reason[:120]}</div>' if reason else ''}
            <div class="text-section">{text_snippet}</div>
          </div>
        </div>"""

        # 每组附上初步诊断提示
        diagnosis = get_diagnosis(pred, human, len(items))

        sections_html += f"""
      <div class="confusion-section" id="{anchor}">
        <div class="section-header">
          <span class="section-badge" style="background:{label_color(pred)}">{pred}</span>
          <span class="section-arrow">预测为</span>
          <span class="section-badge" style="background:{label_color(human)}">{human}</span>
          <span class="section-count">（{len(items)} 个错误）</span>
        </div>
        <div class="diagnosis-box">{diagnosis}</div>
        <div class="cards-grid">{cards_html}</div>
      </div>"""

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>CTFAD — 错误案例分析</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f0f2f5; color: #1a1a1a; }}

.topbar {{
  background: #1a1a2e; color: white; padding: 14px 24px;
  display: flex; align-items: center; justify-content: space-between;
  position: sticky; top: 0; z-index: 100;
  box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}}
.topbar h1 {{ font-size: 16px; font-weight: 600; }}
.topbar .summary {{ font-size: 13px; opacity: 0.8; }}
.accuracy-badge {{
  display: inline-block; padding: 4px 12px; border-radius: 20px;
  font-weight: 700; font-size: 14px;
  background: {"#ef4444" if accuracy < 70 else "#f59e0b" if accuracy < 85 else "#22c55e"};
  color: white; margin-left: 12px;
}}

.container {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}

/* 混淆对导航 */
.confusion-nav {{
  background: white; border-radius: 12px; padding: 16px 20px;
  margin-bottom: 24px; box-shadow: 0 1px 4px rgba(0,0,0,0.08);
}}
.confusion-nav h2 {{ font-size: 14px; color: #666; margin-bottom: 12px; }}
.confusion-pills {{ display: flex; flex-wrap: wrap; gap: 10px; }}
.confusion-pill {{
  display: flex; align-items: center; gap: 6px;
  padding: 6px 12px; border-radius: 20px;
  background: #f8f9fa; border: 1px solid #e9ecef;
  text-decoration: none; color: inherit;
  transition: box-shadow 0.15s;
}}
.confusion-pill:hover {{ box-shadow: 0 2px 8px rgba(0,0,0,0.12); }}
.pill-pred, .pill-human {{
  padding: 2px 8px; border-radius: 10px; color: white;
  font-size: 12px; font-weight: 600;
}}
.pill-arrow {{ font-size: 12px; color: #999; }}
.pill-count {{
  background: #e9ecef; border-radius: 10px;
  padding: 1px 7px; font-size: 12px; font-weight: 700; color: #666;
}}

/* 混淆组 */
.confusion-section {{
  background: white; border-radius: 12px; padding: 20px;
  margin-bottom: 24px; box-shadow: 0 1px 4px rgba(0,0,0,0.08);
}}
.section-header {{
  display: flex; align-items: center; gap: 10px; margin-bottom: 12px;
}}
.section-badge {{
  padding: 4px 12px; border-radius: 20px; color: white;
  font-size: 13px; font-weight: 700;
}}
.section-arrow {{ font-size: 13px; color: #999; }}
.section-count {{ font-size: 13px; color: #888; }}

/* 诊断框 */
.diagnosis-box {{
  background: #fffbeb; border: 1px solid #fde68a; border-radius: 8px;
  padding: 12px 16px; margin-bottom: 16px;
  font-size: 13px; line-height: 1.7; color: #78350f;
}}

/* 卡片网格 */
.cards-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 16px;
}}
.error-card {{
  border: 1px solid #fecaca; border-radius: 10px; overflow: hidden;
  background: #fff5f5;
}}
.card-img {{
  width: 100%; height: 200px; object-fit: contain;
  background: #fafafa; cursor: pointer; border-bottom: 1px solid #fee2e2;
}}
.card-img:hover {{ opacity: 0.85; }}
.card-body {{ padding: 12px 14px; }}
.card-meta {{ font-size: 11px; color: #999; font-family: monospace; margin-bottom: 8px; }}

.label-row {{
  display: flex; align-items: center; gap: 8px; margin-bottom: 8px;
}}
.label-block {{ display: flex; flex-direction: column; gap: 3px; }}
.label-title {{ font-size: 10px; color: #999; }}
.label-badge {{
  display: inline-block; padding: 3px 10px; border-radius: 20px;
  font-size: 12px; font-weight: 600; color: white;
}}
.conf {{ font-size: 11px; color: #aaa; }}
.arrow-block {{ font-size: 16px; color: #ef4444; padding: 0 4px; margin-top: 12px; }}
.wrong-block .label-badge {{ opacity: 0.7; text-decoration: line-through; }}

.reason-row {{
  font-size: 11px; color: #999; font-style: italic;
  margin-bottom: 8px; line-height: 1.4;
}}

.text-section {{ margin-top: 6px; }}
.text-row {{
  font-size: 11px; color: #666; line-height: 1.5;
  border-left: 2px solid #fca5a5; padding-left: 8px; margin-bottom: 4px;
}}
.text-row.empty {{ color: #ccc; border-left-color: #eee; }}
.text-label {{
  display: inline-block; background: #fee2e2; color: #991b1b;
  font-size: 10px; padding: 1px 5px; border-radius: 3px;
  margin-right: 5px; font-weight: 600;
}}

/* 灯箱 */
.lightbox {{
  display: none; position: fixed; inset: 0;
  background: rgba(0,0,0,0.88); z-index: 1000;
  align-items: center; justify-content: center;
}}
.lightbox.active {{ display: flex; }}
.lightbox img {{ max-width: 90vw; max-height: 90vh; object-fit: contain; border-radius: 4px; }}
.lightbox-close {{
  position: absolute; top: 20px; right: 24px;
  color: white; font-size: 28px; cursor: pointer;
}}
</style>
</head>
<body>

<div class="topbar">
  <h1>🔍 CTFAD 错误案例分析</h1>
  <div class="summary">
    错误 {len(errors)} / {total} 条
    <span class="accuracy-badge">{accuracy:.0f}%</span>
  </div>
</div>

<div class="container">

  <div class="confusion-nav">
    <h2>混淆对一览（点击跳转）</h2>
    <div class="confusion-pills">{confusion_nav}</div>
  </div>

  {sections_html}

</div>

<div class="lightbox" id="lightbox" onclick="closeLightbox()">
  <span class="lightbox-close">✕</span>
  <img id="lightboxImg" src="" alt="">
</div>

<script>
function openLightbox(src) {{
  document.getElementById('lightboxImg').src = src;
  document.getElementById('lightbox').classList.add('active');
}}
function closeLightbox() {{
  document.getElementById('lightbox').classList.remove('active');
}}
</script>
</body>
</html>"""


def get_diagnosis(pred, human, count):
    """根据混淆对给出初步诊断提示，帮助判断 prompt 问题还是模型能力问题"""
    diagnoses = {
        ("vessel", "diagram"): """
          <b>初步诊断方向：</b> 模型把「示意图」判成了「花器图」。
          可能原因：①示意图中画有花器轮廓，模型抓住了「器形」特征；②prompt 中 vessel 的描述过于宽泛。
          <br><b>观察重点：</b> 这些图是否画有花器外形？图中是否有箭头/数字标注？
          <br><b>诊断方式：</b> 如果图片清晰且明显是示意图 → prompt 问题（可修复）；如果图片模糊或类别模糊 → 模型能力问题。
        """,
        ("vessel", "arrangement"): """
          <b>初步诊断方向：</b> 模型把「插花作品」判成了「花器图」。
          可能原因：图中花器比较显眼，花材较少或颜色淡，模型焦点落在器皿上。
          <br><b>观察重点：</b> 这些图里花材是否很少/颜色不鲜艳？花器是否占画面很大比例？
          <br><b>诊断方式：</b> 如果花材明显存在 → prompt 问题；如果确实花材极少、作品不完整 → 类别边界模糊。
        """,
        ("diagram", "arrangement"): """
          <b>初步诊断方向：</b> 模型把「插花作品图」判成了「示意图」。
          可能原因：教材中作品图旁边常有图注文字，模型因「有文字」就判成 diagram。
          <br><b>观察重点：</b> 这些图是否有图注文字？图片内部是否有箭头或标注线？
          <br><b>诊断方式：</b> 如果图片本身没有箭头/标注线，只是旁边有文字 → prompt 问题（最容易修复）。
        """,
        ("arrangement", "plants"): """
          <b>初步诊断方向：</b> 模型把「花材图」判成了「插花作品」。
          可能原因：花材图中花枝较多或有部分器皿，模型误判为完成的作品。
          <br><b>观察重点：</b> 这些图是否有花器？花材是否被「插」在某个容器里？
          <br><b>诊断方式：</b> 如果没有花器只有散落花材 → prompt 问题；如果确实有花器但作品未完成 → 类别边界模糊。
        """,
        ("photograph", "arrangement"): """
          <b>初步诊断方向：</b> 模型把「插花作品」判成了「场景/其他照片」。
          可能原因：照片背景复杂（展览场景、室内环境），模型焦点不在插花作品上。
          <br><b>观察重点：</b> 插花作品在画面中是否处于中心位置？背景是否过于复杂？
          <br><b>诊断方式：</b> 如果作品明显居中 → prompt 问题；如果作品偏小且背景复杂 → 模型能力或类别边界问题。
        """,
    }
    key = (pred, human)
    if key in diagnoses:
        return diagnoses[key]
    return f"<b>混淆对 {pred} → {human}：</b> 请观察这些图片，判断模型判错的共同特征。"


if __name__ == "__main__":
    main()
