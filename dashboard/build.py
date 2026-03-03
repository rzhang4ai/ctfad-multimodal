#!/usr/bin/env python3
"""
dashboard/build.py
──────────────────
将 dashboard/data.json 的最新内容嵌入到 dashboard/index.html 的
getEmbeddedData() 函数中，使 GitHub Pages 部署时无需 data.json 也能展示正确数据。

用法（从项目根目录执行）：
    python3 dashboard/build.py

完整发布流程：
    python3 dashboard/build.py
    git add dashboard/index.html
    git commit -m "docs(dashboard): sync embedded data for deployment"
    git push
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent  # dashboard/

DATA_PATH  = ROOT / "data.json"
INDEX_PATH = ROOT / "index.html"

def main():
    if not DATA_PATH.exists():
        print(f"❌ {DATA_PATH} not found")
        sys.exit(1)
    if not INDEX_PATH.exists():
        print(f"❌ {INDEX_PATH} not found")
        sys.exit(1)

    data = json.load(open(DATA_PATH, encoding="utf-8"))
    data_str = json.dumps(data, ensure_ascii=False, indent=2)

    html = INDEX_PATH.read_text(encoding="utf-8")

    new_func = (
        "// ═══════════════════════════════════════════════════════\n"
        "//  Embedded fallback data (in case data.json can't be fetched)\n"
        "// ═══════════════════════════════════════════════════════\n"
        "function getEmbeddedData() {\n"
        "  // __EMBEDDED_DATA_BEGIN__\n"
        f"  return {data_str};\n"
        "  // __EMBEDDED_DATA_END__\n"
        "}"
    )

    pattern = (
        r'// ═+\n//  Embedded fallback data.*?// ═+\n'
        r'function getEmbeddedData\(\) \{.*?// __EMBEDDED_DATA_END__\s*\n\}'
    )
    # Use a lambda so re.sub doesn't process \n in new_func as literal newlines
    new_html, n = re.subn(pattern, lambda m: new_func, html, flags=re.DOTALL)

    if n == 0:
        print("❌ Could not find getEmbeddedData() marker in index.html")
        sys.exit(1)

    INDEX_PATH.write_text(new_html, encoding="utf-8")

    stages = data.get("stages", [])
    stage_summary = " | ".join(
        f"{s['label']} {s.get('progress',0)}%" for s in stages
    )
    print(f"✅ index.html updated: {stage_summary}")
    print(f"   lastUpdated: {data['meta'].get('lastUpdated','—')}")
    print(f"   Ready to: git add dashboard/index.html && git commit && git push")

if __name__ == "__main__":
    main()
