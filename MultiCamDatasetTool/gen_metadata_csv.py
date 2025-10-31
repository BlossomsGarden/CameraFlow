# Qwen——————
# conda activate wlh-py
# cd /data2/wlh/waymo-long-tail-5000/paper
# CUDA_VISIBLE_DEVICES=0  python infer.py
# 
# Gen Metadata CSV ——————
# conda activate wlh-py
# cd /data/wlh/ReCamMaster/MultiCamVideo-Dataset
# CUDA_VISIBLE_DEVICES=0  python gen_metadata_csv.py
#
# 注意手动改一下里面可能出现全是感叹号的情况
#
# 不必描述摄像头移动了，因为它理解不了，全都是stationary我生成个毛线

import torch
import os
import csv
from pathlib import Path
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from awq import AutoAWQForCausalLM
import glob

def process_video_with_caption(model, processor, video_path):
    """
    处理单个视频并生成caption
    """
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": 768 * 768,
                        "fps": 1.0,
                    },
                    {
                        "type": "text", 
                        "text": "Briefly describe the content of this video."
                    }
                ],
            }
        ]

        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = inputs.to("cuda")

        # Inference
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text[0].strip()
    
    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")
        return f"Error: {str(e)}"


def clean_csv_encoding(csv_path):
    """
    清理CSV文件中的编码问题
    """
    try:
        # 读取原始文件内容
        with open(csv_path, 'rb') as f:
            content = f.read()
        
        # 尝试用不同编码解码
        try:
            # 首先尝试UTF-8
            decoded_content = content.decode('utf-8')
        except UnicodeDecodeError:
            try:
                # 如果UTF-8失败，尝试其他常见编码
                decoded_content = content.decode('latin-1')
                print("Warning: Used latin-1 encoding to fix file")
            except:
                # 最后尝试忽略错误
                decoded_content = content.decode('utf-8', errors='ignore')
                print("Warning: Used utf-8 with errors ignored to fix file")
        
        # 清理非法字符
        cleaned_content = ''
        for char in decoded_content:
            # 只保留可打印的ASCII字符和常见Unicode字符
            if ord(char) >= 32 or char in '\t\n\r':
                cleaned_content += char
            else:
                # 替换非法字符为空格
                cleaned_content += ' '
                print(f"Replaced illegal character: {ord(char)}")
        
        # 写回文件
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
        
        print(f"Successfully cleaned CSV file: {csv_path}")
        
    except Exception as e:
        print(f"Error cleaning CSV file: {str(e)}")


def validate_csv_format(csv_path):
    """
    验证CSV文件格式，确保所有file_name都是字符串，并报告空行位置
    """
    try:
        import pandas as pd
        # 读取CSV文件
        df = pd.read_csv(csv_path)
        
        # # 检查file_name列的数据类型
        # print(f"Data types in CSV:")
        # print(df.dtypes)

        # 记录空行位置
        empty_rows = []
        # 检查每一行，记录空值或非字符串值
        for idx, row in df.iterrows():
            file_name = row['file_name']
            # 检查是否为NaN或空字符串
            if pd.isna(file_name) or file_name == '':
                empty_rows.append(idx + 2)  # +2 是因为索引从0开始，且跳过标题行
                print(f"Warning: Empty file_name at line {idx + 2}")
            # 检查是否为非字符串类型
            elif not isinstance(file_name, str):
                print(f"Warning: Non-string file_name at line {idx + 2}: {type(file_name)} -> {file_name}")
        
        # 确保file_name列是字符串类型
        if df['file_name'].dtype != object:  # object通常表示字符串
            print(f"Converting file_name column from {df['file_name'].dtype} to string")
            df['file_name'] = df['file_name'].astype(str)
        
        # 检查是否有NaN或空值
        nan_count = df['file_name'].isna().sum()
        if nan_count > 0:
            print(f"Found {nan_count} NaN values in file_name column, filling with empty string")
            df['file_name'] = df['file_name'].fillna('')
        
        # 移除可能存在的空行
        original_rows = len(df)
        df = df.dropna(how='all')  # 删除全为空的行
        removed_count = original_rows - len(df)
        if removed_count > 0:
            print(f"Removed {removed_count} empty rows")
        
        # 保存修复后的CSV
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"CSV format validated and fixed: {csv_path}")
        
        # 报告空行情况
        if empty_rows:
            print(f"Found {len(empty_rows)} empty rows at lines: {empty_rows}")
        else:
            print("No empty rows found in the CSV file.")
        
        return True
        
    except Exception as e:
        print(f"Error validating CSV format: {str(e)}")
        return False

def main():
    # 配置参数
    dataset_dir = "/data/wlh/ReCamMaster/MultiCamVideo-Dataset/MultiCamVideo-Dataset/"
    dataset_type = "train"
    dataset_cam_type = "f18_aperture10"
    
    # 最终输出文件路径
    output_csv_path = os.path.join(dataset_dir, "metadata.csv")
    
    # 加载模型和处理器
    model_path = '/data2/wlh/waymo-long-tail-5000/paper/Qwen2.5-VL-3B-Instruct-AWQ'
    print("Loading model...")
    model = AutoAWQForCausalLM.from_quantized(model_path)
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(model_path)
    print("Model and processor loaded successfully!")
    
    # 构建基础路径
    base_path = os.path.join(dataset_dir, dataset_type, dataset_cam_type)
    
    # 获取所有scene文件夹
    scene_pattern = os.path.join(base_path, "scene*")
    scene_folders = sorted(glob.glob(scene_pattern))
    
    
    print(f"Found {len(scene_folders)} scene folders")
    
    # 收集所有视频文件
    all_video_data = []
    for scene_folder in scene_folders:
        # 获取所有mp4文件
        scene_name = os.path.basename(scene_folder)
        videos_dir = os.path.join(scene_folder, "videos")
        video_files = sorted(glob.glob(os.path.join(videos_dir, "*.mp4")))
        print(f"Found {len(video_files)} videos in {scene_name}")
        
        # 为每个视频创建相对路径
        for video_path in video_files:
            # 计算相对于dataset_dir的相对路径
            relative_path = os.path.relpath(video_path, os.path.join(dataset_dir, dataset_type))
            all_video_data.append((video_path, relative_path))
    
    print(f"Total videos to process: {len(all_video_data)}")
    
    # 创建CSV文件
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['file_name', 'text'])
        
        # 处理每个视频
        for i, (video_path, relative_path) in enumerate(all_video_data, 1):
            print(f"Processing video {i}/{len(all_video_data)}: {relative_path}")
            
            # 生成caption
            caption = process_video_with_caption(model, processor, video_path)
            caption = caption.encode('utf-8', errors='replace').decode('utf-8')
            
            # 写入CSV（使用相对路径）
            writer.writerow([relative_path, caption])
            print(f"Generated caption for {relative_path}: {caption[:100]}...")  # 只打印前100个字符
            
            # 每处理10个视频就flush一次，确保数据及时写入
            if i % 10 == 0:
                csvfile.flush()
                print(f"Progress: {i}/{len(all_video_data)} videos processed")
    
    print(f"All videos processed! Metadata file created: {output_csv_path}")


    print("Cleaning CSV file encoding...")
    clean_csv_encoding(output_csv_path)
    print("CSV file cleaning completed!")
    
    
    print("Validating CSV format...")
    validate_csv_format(output_csv_path)
    print("CSV format validation completed!")

if __name__ == "__main__":
    main()