import os
import json
from pathlib import Path
from PIL import Image
from rembg import remove
from tqdm import tqdm

# 配置路径
SUBSET_DIR = Path("/Volumes/aiworkbench/datasets/06_classified/subsets_arrangement/arrangement_photo")
INPUT_INDEX = SUBSET_DIR / "index_profiled.jsonl"    # 读取我们刚刚打完分的索引
OUTPUT_INDEX = SUBSET_DIR / "index_with_rmbg.jsonl"  # 写入包含了去背图片路径的新索引

IMG_OUT_DIR = SUBSET_DIR / "images_rmbg"

def main():
    if not os.path.exists(INPUT_INDEX):
        print(f"找不到输入文件: {INPUT_INDEX}")
        return

    os.makedirs(IMG_OUT_DIR, exist_ok=True)
    
    records =[]
    with open(INPUT_INDEX, 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f]
        
    # 统计一下有多少 A 级图片要处理
    target_count = sum(1 for r in records if r.get("quality_profile", {}).get("quality_tier") == "A")
    print(f"找到 {target_count} 张 A 级高质量图片，准备进行背景去除...")
    print(f"(M4 芯片纯 CPU 跑 rembg，单张大约需 2-5 秒，请耐心等待)\n")
    
    updated_records =[]
    for record in tqdm(records, desc="Processing Images"):
        # 提取评估等级
        tier = record.get("quality_profile", {}).get("quality_tier")
        src_path = record.get("subset_image_path")
        
        # 只处理 A 级图片
        if tier == "A" and src_path and os.path.exists(src_path):
            filename = os.path.basename(src_path)
            base_name = os.path.splitext(filename)[0]
            out_path = IMG_OUT_DIR / f"{base_name}_rgba.png"
            
            # 断点续传：如果已经抠过，直接跳过
            if not os.path.exists(out_path):
                try:
                    # 读取原图并使用 rembg 去背景
                    input_img = Image.open(src_path)
                    output_img = remove(input_img)
                    output_img.save(out_path, "PNG")
                except Exception as e:
                    print(f"\n处理 {filename} 失败: {e}")
                    # 如果失败，保留原记录但不加 rmbg 路径
                    updated_records.append(record)
                    continue
                    
            # 记录抠图后的绝对路径
            record["rmbg_image_path"] = str(out_path)
        
        updated_records.append(record)
        
    # 将更新后的数据写回新索引
    with open(OUTPUT_INDEX, 'w', encoding='utf-8') as f:
        for r in updated_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    print(f"\n✅ 背景去除完成！")
    print(f"透明背景 PNG 已保存在: {IMG_OUT_DIR}")
    print(f"最新索引数据已更新至: index_with_rmbg.jsonl")

if __name__ == "__main__":
    main()