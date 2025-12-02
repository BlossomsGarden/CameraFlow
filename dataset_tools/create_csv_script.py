#!/usr/bin/env python3
# 批量创建32个脚本文件

template = """
import torch
import os
import csv
from pathlib import Path
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import glob

def process_video_with_caption(model, processor, video_path):
    \"\"\"
    处理单个视频并生成caption
    \"\"\"
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "fps": 30.0,
                    },
                    {
                        "type": "text", 
                        "text": "Descibe this video. Avoid describing text, language, or camera movement in the scene. Do not use paragraphs or bullet points, provide a few short sentences directly. Do not use speculative terms like 'appear to' or 'seem'."
                    }
                ],
            }
        ]

        # Preparation for inference
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)

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


def main():
    # 配置参数
    dataset_dir = "/home/ma-user/modelarts/user-job-dir/wlh/Data/MultiCamVideo-Dataset/"
    dataset_type = "train"
    dataset_cam_type = "f24_aperture5"
    
    # 最终输出文件路径
    output_csv_path = os.path.join(dataset_dir, f"metadata-{dataset_cam_type}-{START}-{END}.csv")
    
    # 加载模型和处理器
    model_path = '/home/ma-user/modelarts/user-job-dir/wlh/Model/Qwen/Qwen3-VL-8B-Instruct'  # 请根据实际路径修改
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    
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
    all_video_data = all_video_data[{START}:{END}]
    
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


if __name__ == "__main__":
    main()
"""

ranges = [
    (0, 1062),
    (1062, 2124),
    (2124, 3186),
    (3186, 4248),
    (4248, 5310),
    (5310, 6372),
    (6372, 7434),
    (7434, 8496),
    (8496, 9558),
    (9558, 10620),
    (10620, 11682),
    (11682, 12744),
    (12744, 13806),
    (13806, 14868),
    (14868, 15930),
    (15930, 16992),
    (16992, 18054),
    (18054, 19116),
    (19116, 20178),
    (20178, 21240),
    (21240, 22302),
    (22302, 23364),
    (23364, 24426),
    (24426, 25488),
    (25488, 26550),
    (26550, 27612),
    (27612, 28674),
    (28674, 29736),
    (29736, 30798),
    (30798, 31860),
    (31860, 32922),
    (32922, 33990),
]
print(f'Creating {len(ranges)} files...')
for start, end in ranges:
    filename = f'gen_metadata_qwen3_{start}_{end}.py'
    content = template.replace('{START}', str(start)).replace('{END}', str(end))
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f'Created {filename}')

print('All files created!')

