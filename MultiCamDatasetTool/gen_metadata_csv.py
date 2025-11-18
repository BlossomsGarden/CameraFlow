# Qwen——————
# conda activate wlh-py
# cd /data2/wlh/waymo-long-tail-5000/paper
# CUDA_VISIBLE_DEVICES=0  python infer.py
# 
# Gen Metadata CSV ——————
# conda activate wlh-py
# cd /data/wlh/ReCamMaster/MultiCamVideo-Dataset
# CUDA_VISIBLE_DEVICES=2 nohup  python gen_metadata_csv-30000-0.py >> 30000-0.out 2>&1 &
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
                        "text": "Provide a concise overview of the video in about 20 words. Avoid describing text, background, style, language, lighting, camera, or other objects in the sene. Do not use paragraphs or bullet points, provide a few short sentences directly. Describe only the scene and human actions objectively (Do not use speculative terms like 'appear to' or 'seem'). Not that there is only one human in the scene."
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


def main():
    # 配置参数
    dataset_dir = "/data/wlh/ReCamMaster/MultiCamVideo-Dataset/MultiCamVideo-Dataset/"
    dataset_type = "train"
    dataset_cam_type = "f50_aperture2.4"
    
    # 最终输出文件路径
    output_csv_path = os.path.join(dataset_dir, f"metadata-{dataset_cam_type}-30000-0.csv")
    
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
    all_video_data = all_video_data[30000:]
    
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