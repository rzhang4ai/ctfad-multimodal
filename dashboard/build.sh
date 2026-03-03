#!/bin/bash
# dashboard/build.sh — 一键更新看板并准备发布
# 用法（从项目根目录执行）：bash dashboard/build.sh
# 或带 push 参数自动推送：  bash dashboard/build.sh --push

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "📦 Embedding data.json into index.html..."
python3 dashboard/build.py

if [ "$1" = "--push" ]; then
    echo ""
    echo "🚀 Staging and pushing index.html..."
    git add dashboard/index.html
    git commit -m "docs(dashboard): sync embedded data $(date +%Y-%m-%d)"
    git push
    echo "✅ Dashboard deployed to GitHub Pages!"
else
    echo ""
    echo "⬆️  To deploy to GitHub Pages, run:"
    echo "   git add dashboard/index.html"
    echo "   git commit -m 'docs(dashboard): sync embedded data'"
    echo "   git push"
    echo ""
    echo "   Or: bash dashboard/build.sh --push"
fi
