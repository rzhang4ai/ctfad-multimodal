import os
import json
import shutil
from pathlib import Path

# 配置路径
INPUT_FILE = "/Volumes/aiworkbench/datasets/06_classified/reviewed/classified_final_fixed.jsonl"
# 使用你修改后的新目录
OUTPUT_BASE_DIR = "/Volumes/aiworkbench/datasets/06_classified/subsets_arrangement"

# 需要提取的目标子集
TARGET_SUBSETS =[
    "arrangement_photo",
    "arrangement_painting",
    "arrangement_drawing"
]

def get_image_path(record):
    """鲁棒的图片路径获取器，兼容不同阶段生成的 JSON 结构"""
    # 1. 如果是嵌套字典 record["image"]["absolute_path"]
    if "image" in record and isinstance(record["image"], dict):
        path = record["image"].get("absolute_path") or record["image"].get("path")
        if path: return path
        
    # 2. 尝试所有可能的平铺键名
    possible_keys =[
        "image.absolute_path", 
        "image_absolute_path", 
        "absolute_path", 
        "image_path", 
        "image"
    ]
    for key in possible_keys:
        if key in record and isinstance(record[key], str):
            return record[key]
            
    return None

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"找不到输入文件: {INPUT_FILE}")
        return

    # 初始化输出目录结构
    subset_files = {}
    for subset in TARGET_SUBSETS:
        subset_dir = Path(OUTPUT_BASE_DIR) / subset
        img_dir = subset_dir / "images"
        os.makedirs(img_dir, exist_ok=True)
        
        # 打开对应的 index.jsonl 文件准备写入
        index_path = subset_dir / "index.jsonl"
        subset_files[subset] = open(index_path, 'w', encoding='utf-8')

    counts = {subset: 0 for subset in TARGET_SUBSETS}
    missing_images = 0

    # 遍历总表并分发
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            fine_label = record.get("fine_label")
            
            if fine_label in TARGET_SUBSETS:
                # 获取原图路径
                src_img_path = get_image_path(record)
                
                if not src_img_path or not os.path.exists(src_img_path):
                    gid = record.get("global_pair_id", "Unknown")
                    print(f"⚠️ 警告：跳过数据 {gid}，找不到有效图片路径或文件不存在 -> {src_img_path}")
                    missing_images += 1
                    continue

                # 构建新图路径并复制
                img_filename = os.path.basename(src_img_path)
                subset_dir = Path(OUTPUT_BASE_DIR) / fine_label
                dst_img_path = subset_dir / "images" / img_filename
                
                # 只有在目标文件不存在时才复制（支持重复运行）
                if not os.path.exists(dst_img_path):
                    shutil.copy2(src_img_path, dst_img_path)
                
                # 更新记录中的图片路径为子集内的绝对路径
                record["subset_image_path"] = str(dst_img_path)
                
                # 写入对应的 index 文件
                subset_files[fine_label].write(json.dumps(record, ensure_ascii=False) + "\n")
                counts[fine_label] += 1

    # 关闭所有文件
    for f in subset_files.values():
        f.close()

    print("\n✅ 子集提取完成！分布如下：")
    for subset, count in counts.items():
        print(f" - {subset}: {count} 条")
        
    if missing_images > 0:
        print(f"\n⚠️ 共有 {missing_images} 条数据因为找不到图片被跳过。")

if __name__ == "__main__":
    main()