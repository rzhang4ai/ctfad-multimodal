#!/usr/bin/env python3
"""
s3_remove_background.py
CTFAD S3 v1 — Golden Test Set 去背景脚本（birefnet-general）

支持三种使用方式：
  方式 1：处理整个目录的所有图片
  方式 2：手动指定图片列表（在脚本中编辑 MANUAL_IMAGES）
  方式 3：命令行参数指定图片路径

运行示例：
    conda activate ctfad-s3-exp  (或 ctfad-s3)

    # 方式1：处理整个 Golden Test Set
    python s3_remove_background.py --mode dir

    # 方式2：处理手动列表
    python s3_remove_background.py --mode manual

    # 方式3：命令行指定
    python s3_remove_background.py --mode files --files /path/to/img1.jpg

输出：
    datasets/07_golden_test_set/images_rmbg/
    文件名：{原文件名}_rgba.png（RGBA 格式，透明背景）

v0 原始版本保留在 s3_experiments_v0/s3_remove_background.py
"""

import os
import sys
import time
import argparse
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
#  ★ 方式2：在这里手动填写要处理的图片路径 ★
#  支持绝对路径 或 相对于 BASE_INPUT_DIR 的文件名
# ═══════════════════════════════════════════════════════════════
MANUAL_IMAGES = [
    # 在此填写 Golden Test Set 选定的图片文件名
    # 等 Golden Test Set 选图完成后更新
]

# ═══════════════════════════════════════════════════════════════
#  路径配置
# ═══════════════════════════════════════════════════════════════
# 默认输入目录（方式1 的扫描目录，方式2 的文件名补全目录）
BASE_INPUT_DIR = Path(
    "/Volumes/aiworkbench/datasets/07_golden_test_set/images"
)

# 输出目录（默认与输入同级，自动创建）
# 设为 None 则自动在输入图片同目录下建 images_rmbg_birefnet/
OUTPUT_DIR = Path("/Volumes/aiworkbench/datasets/07_golden_test_set/images_rmbg")

# ═══════════════════════════════════════════════════════════════
#  去背景参数配置
# ═══════════════════════════════════════════════════════════════
REMBG_CONFIG = {
    "model":                      "birefnet-general",
    "alpha_matting":              True,
    "alpha_matting_foreground_threshold": 240,
    "alpha_matting_background_threshold": 10,
    "alpha_matting_erode_size":   10,
}

# 三条铁律参数（防止 OOM）
SLEEP_BETWEEN   = 1.5   # 每张图之间等待（秒）
SLEEP_EVERY_N   = 10    # 每 N 张额外休息
SLEEP_EXTRA     = 8.0   # 额外休息时长（秒）

# 图片预处理：是否先压缩到最大边长（None = 不压缩，保留原始尺寸）
MAX_EDGE_SIZE = None     # 例：1024 → 压缩到最长边 1024px；None = 原图尺寸

# ═══════════════════════════════════════════════════════════════


def check_dependencies():
    """检查必要依赖"""
    missing = []
    try:
        from rembg import remove, new_session
    except ImportError:
        missing.append("rembg")
    try:
        from PIL import Image
    except ImportError:
        missing.append("pillow")

    if missing:
        print("❌ 缺少依赖，请运行：")
        print(f"   pip install {' '.join(missing)} --break-system-packages")
        sys.exit(1)


def get_output_dir(input_path: Path) -> Path:
    """决定输出目录"""
    if OUTPUT_DIR is not None:
        d = Path(OUTPUT_DIR)
    else:
        d = input_path.parent.parent / "images_rmbg_birefnet"
        # 如果输入文件直接在 BASE_INPUT_DIR 根下，输出放在同级的 images_rmbg_birefnet/
        if input_path.parent == BASE_INPUT_DIR:
            d = BASE_INPUT_DIR.parent / "images_rmbg_birefnet"
        else:
            d = input_path.parent.parent / "images_rmbg_birefnet"
    d.mkdir(parents=True, exist_ok=True)
    return d


def compress_if_needed(img, max_edge: int):
    """可选：压缩图片到最大边长"""
    if max_edge is None:
        return img
    w, h = img.size
    if max(w, h) <= max_edge:
        return img
    ratio = max_edge / max(w, h)
    new_w, new_h = int(w * ratio), int(h * ratio)
    from PIL import Image
    return img.resize((new_w, new_h), Image.LANCZOS)


def process_image(img_path: Path, session, verbose: bool = True) -> dict:
    """对单张图片执行去背景，返回结果信息"""
    from rembg import remove
    from PIL import Image

    result = {
        "input": str(img_path),
        "output": None,
        "success": False,
        "error": None,
        "input_size": None,
        "output_size": None,
        "elapsed_sec": None,
    }

    if not img_path.exists():
        result["error"] = "文件不存在"
        return result

    out_dir = get_output_dir(img_path)
    out_filename = img_path.stem + "_rgba.png"
    out_path = out_dir / out_filename

    # 断点续传：已存在则跳过
    if out_path.exists():
        if verbose:
            print(f"   ⏭  已存在，跳过：{out_filename}")
        result["output"] = str(out_path)
        result["success"] = True
        result["skipped"] = True
        return result

    try:
        t0 = time.time()

        # 读取并可选压缩
        img = Image.open(img_path).convert("RGBA" if img_path.suffix.lower() == ".png" else "RGB")
        orig_size = img.size
        img = compress_if_needed(img, MAX_EDGE_SIZE)
        result["input_size"] = f"{orig_size[0]}×{orig_size[1]}"
        if img.size != orig_size:
            result["input_size"] += f" → {img.size[0]}×{img.size[1]}（已压缩）"

        # 去背景
        output_img = remove(
            img,
            session=session,
            alpha_matting=REMBG_CONFIG["alpha_matting"],
            alpha_matting_foreground_threshold=REMBG_CONFIG["alpha_matting_foreground_threshold"],
            alpha_matting_background_threshold=REMBG_CONFIG["alpha_matting_background_threshold"],
            alpha_matting_erode_size=REMBG_CONFIG["alpha_matting_erode_size"],
        )

        # 保存
        output_img.save(out_path, "PNG")
        elapsed = time.time() - t0

        result["output"] = str(out_path)
        result["output_size"] = f"{output_img.size[0]}×{output_img.size[1]}"
        result["success"] = True
        result["elapsed_sec"] = round(elapsed, 1)

        if verbose:
            print(f"   ✅ {out_filename}  ({result['input_size']}, {elapsed:.1f}s)")

    except Exception as e:
        result["error"] = str(e)
        if verbose:
            print(f"   ❌ 失败：{e}")

    return result


def resolve_image_list(raw_list: list) -> list:
    """
    将 MANUAL_IMAGES 中的条目解析为实际 Path 对象。
    - 绝对路径：直接用
    - 文件名（无 /）：在 BASE_INPUT_DIR 下查找
    """
    paths = []
    for item in raw_list:
        item = item.strip()
        if not item:
            continue
        p = Path(item)
        if p.is_absolute():
            paths.append(p)
        else:
            # 当作文件名，在 BASE_INPUT_DIR 下找
            candidate = BASE_INPUT_DIR / item
            paths.append(candidate)
    return paths


def collect_dir_images(directory: Path) -> list:
    """收集目录下所有图片"""
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
    return sorted([
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    ])


def main():
    check_dependencies()
    from rembg import new_session

    parser = argparse.ArgumentParser(description="CTFAD S3 去背景工具")
    parser.add_argument(
        "--mode", choices=["manual", "dir", "files"], default="manual",
        help="manual=手动列表 | dir=整个目录 | files=命令行指定文件"
    )
    parser.add_argument(
        "--files", nargs="+", metavar="PATH",
        help="[mode=files 时使用] 图片路径列表"
    )
    parser.add_argument(
        "--dir", metavar="DIR", default=None,
        help="[mode=dir 时使用] 覆盖 BASE_INPUT_DIR"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("CTFAD S3 — 去背景处理（birefnet-general）")
    print("=" * 60)

    # ── 收集要处理的图片 ──────────────────────
    if args.mode == "manual":
        if not MANUAL_IMAGES:
            print("❌ MANUAL_IMAGES 列表为空！")
            print("   请在脚本顶部的 MANUAL_IMAGES = [...] 中填写图片路径。")
            sys.exit(1)
        img_paths = resolve_image_list(MANUAL_IMAGES)
        print(f"📋 模式：手动列表，共 {len(img_paths)} 张")

    elif args.mode == "dir":
        scan_dir = Path(args.dir) if args.dir else BASE_INPUT_DIR
        img_paths = collect_dir_images(scan_dir)
        print(f"📂 模式：目录扫描  {scan_dir}")
        print(f"   找到 {len(img_paths)} 张图片")

    elif args.mode == "files":
        if not args.files:
            print("❌ --mode files 需要通过 --files 指定图片路径")
            sys.exit(1)
        img_paths = [Path(f) for f in args.files]
        print(f"📎 模式：命令行指定，共 {len(img_paths)} 张")

    if not img_paths:
        print("❌ 没有找到可处理的图片")
        sys.exit(1)

    # 显示待处理列表
    print()
    print("待处理图片：")
    for p in img_paths:
        status = "✅" if p.exists() else "❌ 不存在"
        print(f"  {status} {p}")

    print()
    print("去背景参数：")
    for k, v in REMBG_CONFIG.items():
        print(f"  {k}: {v}")
    if MAX_EDGE_SIZE:
        print(f"  max_edge_size: {MAX_EDGE_SIZE}px")
    print()

    # ── 加载模型 ──────────────────────────────
    print(f"⏳ 加载模型 {REMBG_CONFIG['model']}（首次运行会自动下载，约 200MB）...")
    try:
        session = new_session(REMBG_CONFIG["model"])
        print("✅ 模型加载完成\n")
    except Exception as e:
        print(f"❌ 模型加载失败：{e}")
        print("   请确认 rembg 已安装：pip install rembg --break-system-packages")
        sys.exit(1)

    # ── 批量处理 ──────────────────────────────
    results = []
    for i, img_path in enumerate(img_paths, 1):
        print(f"[{i:2d}/{len(img_paths)}] {img_path.name}")
        r = process_image(img_path, session)
        results.append(r)

        # 铁律1：冷却
        if i < len(img_paths):
            time.sleep(SLEEP_BETWEEN)
            if i % SLEEP_EVERY_N == 0:
                print(f"   💤 每 {SLEEP_EVERY_N} 张额外休息 {SLEEP_EXTRA}s...")
                time.sleep(SLEEP_EXTRA)

    # ── 汇总 ──────────────────────────────────
    print()
    print("=" * 60)
    succeeded = [r for r in results if r.get("success") and not r.get("skipped")]
    skipped   = [r for r in results if r.get("skipped")]
    failed    = [r for r in results if not r.get("success")]

    print(f"✅ 成功：{len(succeeded)} 张")
    print(f"⏭  跳过（已存在）：{len(skipped)} 张")
    if failed:
        print(f"❌ 失败：{len(failed)} 张")
        for r in failed:
            print(f"   · {r['input']}  →  {r.get('error','未知错误')}")

    if succeeded:
        out_dir_example = Path(succeeded[0]["output"]).parent
        print(f"\n📁 输出目录：{out_dir_example}")
        print("   文件列表：")
        for r in succeeded:
            print(f"   · {Path(r['output']).name}")

    print("=" * 60)


if __name__ == "__main__":
    main()
