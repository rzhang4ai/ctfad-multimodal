import os
import json
from pathlib import Path
from PIL import Image
from rembg import remove, new_session
from tqdm import tqdm

# 配置路径
SUBSET_DIR = Path("/Volumes/aiworkbench/datasets/06_classified/subsets_arrangement/arrangement_photo")
INPUT_INDEX = SUBSET_DIR / "index_profiled.jsonl"    
OUTPUT_INDEX = SUBSET_DIR / "index_with_rmbg_isnet.jsonl"  

# 新建一个专属的高精度输出目录
IMG_OUT_DIR = SUBSET_DIR / "images_rmbg_isnet"

def main():
    if not os.path.exists(INPUT_INDEX):
        print(f"找不到输入文件: {INPUT_INDEX}")
        return

    os.makedirs(IMG_OUT_DIR, exist_ok=True)
    
    # 核心修改点：加载高精度细节模型 (IS-Net)
    print("正在加载高精度细节模型 birefnet-general (2026 年，现在的 rembg 其实还内置了最新一代的 birefnet-general 模型，它对极细的网格、头发丝、甚至半透明的植物绒毛提取能力比 isnet 还要变态)...")
    isnet_session = new_session("birefnet-general")
    
    records =[]
    with open(INPUT_INDEX, 'r', encoding='utf-8') as f:
        records =[json.loads(line) for line in f]
        
    target_count = sum(1 for r in records if r.get("quality_profile", {}).get("quality_tier") == "A")
    print(f"找到 {target_count} 张 A 级图片，准备重新进行高精度抠图...\n")
    
    updated_records =[]
    for record in tqdm(records, desc="Processing Images"):
        tier = record.get("quality_profile", {}).get("quality_tier")
        src_path = record.get("subset_image_path")
        
        if tier == "A" and src_path and os.path.exists(src_path):
            filename = os.path.basename(src_path)
            base_name = os.path.splitext(filename)[0]
            out_path = IMG_OUT_DIR / f"{base_name}_rgba.png"
            
            if not os.path.exists(out_path):
                try:
                    input_img = Image.open(src_path)
                    
                    # 使用 isnet 模型进行高精度去背
                    # 开启 alpha_matting 进一步保护边缘半透明像素
                    output_img = remove(
                        input_img, 
                        session=isnet_session,
                        alpha_matting=True,
                        alpha_matting_foreground_threshold=240,
                        alpha_matting_background_threshold=10,
                        alpha_matting_erode_size=10
                    )
                    output_img.save(out_path, "PNG")
                except Exception as e:
                    print(f"\n处理 {filename} 失败: {e}")
                    updated_records.append(record)
                    continue
                    
            record["rmbg_image_path"] = str(out_path)
        
        updated_records.append(record)
        
    with open(OUTPUT_INDEX, 'w', encoding='utf-8') as f:
        for r in updated_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    print(f"\n✅ 高精度背景去除完成！")
    print(f"请前往 {IMG_OUT_DIR} 查看枝条是否恢复！")

if __name__ == "__main__":
    main()