import os
import json
import cv2
from pathlib import Path
from tqdm import tqdm

# 配置路径 (请确保路径正确)
SUBSET_DIR = Path("/Volumes/aiworkbench/datasets/06_classified/subsets_arrangement/arrangement_photo")
INPUT_INDEX = SUBSET_DIR / "index.jsonl"
OUTPUT_INDEX = SUBSET_DIR / "index_profiled.jsonl"

IMG_DIR = SUBSET_DIR / "images"
COMPLICATED_DIR = SUBSET_DIR / "images" / "complicated"

def analyze_cv_metrics(img_path):
    """使用 OpenCV 计算客观物理指标"""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    
    h, w, _ = img.shape
    
    # 模糊度：拉普拉斯方差 (越低越模糊，通常 <100 算模糊)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 色彩饱和度：判断是否为黑白图片
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].mean()
    is_grayscale = bool(saturation < 10) # 饱和度极低视为黑白
    
    return {
        "width": w,
        "height": h,
        "blur_variance": round(blur_score, 2),
        "is_grayscale": is_grayscale
    }

def main():
    if not os.path.exists(INPUT_INDEX):
        print(f"找不到输入文件: {INPUT_INDEX}")
        return

    records =[]
    with open(INPUT_INDEX, 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f]

    print(f"开始同步手工分类并分析图像质量 (共 {len(records)} 条)...")
    
    stats = {"A_ready": 0, "B_grayscale_or_blur": 0, "C_manual_complicated": 0, "Missing": 0}
    updated_records =[]

    for record in tqdm(records):
        # 获取图片原文件名
        img_path_str = record.get("subset_image_path", "")
        if not img_path_str:
            filename = os.path.basename(record.get("image", {}).get("absolute_path", ""))
        else:
            filename = os.path.basename(img_path_str)

        if not filename:
            stats["Missing"] += 1
            continue

        # 判断图片当前所在的真实位置
        good_path = IMG_DIR / filename
        complicated_path = COMPLICATED_DIR / filename

        quality_profile = {
            "cv_metrics": None,
            "quality_tier": "Unknown",
            "reasons":[]
        }

        if complicated_path.exists():
            # 1. 发现被你手工移入 complicated 文件夹的图片
            record["subset_image_path"] = str(complicated_path)
            quality_profile["quality_tier"] = "C"
            quality_profile["reasons"].append("Manual filtered: Complicated or Truncated")
            stats["C_manual_complicated"] += 1

        elif good_path.exists():
            # 2. 留在外面的好图片，进行 OpenCV 物理指标检测
            record["subset_image_path"] = str(good_path)
            cv_metrics = analyze_cv_metrics(good_path)
            quality_profile["cv_metrics"] = cv_metrics
            
            if cv_metrics:
                is_blur = cv_metrics["blur_variance"] < 100
                is_gray = cv_metrics["is_grayscale"]
                
                if is_blur or is_gray:
                    quality_profile["quality_tier"] = "B"
                    if is_blur: quality_profile["reasons"].append("Low Sharpness (Blurry)")
                    if is_gray: quality_profile["reasons"].append("Grayscale Image")
                    stats["B_grayscale_or_blur"] += 1
                else:
                    quality_profile["quality_tier"] = "A"
                    stats["A_ready"] += 1
        else:
            print(f"⚠️ 找不到文件: {filename}")
            stats["Missing"] += 1
            continue

        record["quality_profile"] = quality_profile
        updated_records.append(record)

    # 写入带有质量评估的新文件
    with open(OUTPUT_INDEX, 'w', encoding='utf-8') as f:
        for r in updated_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n✅ 分析完成！质量分布如下：")
    print(f" - [A级] 完美可直接用于 3D: {stats['A_ready']} 张")
    print(f" - [B级] 物理质量欠佳(黑白/模糊): {stats['B_grayscale_or_blur']} 张")
    print(f" - [C级] 手工过滤(复杂/残缺): {stats['C_manual_complicated']} 张")
    if stats["Missing"] > 0:
        print(f" - [丢失] 找不到文件: {stats['Missing']} 张")

if __name__ == "__main__":
    main()