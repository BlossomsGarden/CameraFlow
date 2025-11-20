#!/usr/bin/env python3
# CUDA_VISIBLE_DEVICES=1 python evaluator.py
# 反正我极度怀疑原论文的 Table 1那里结果乘了100
#!/usr/bin/env python3
import os
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import json
from typing import List, Dict, Tuple
import tempfile
import subprocess
import pickle

from diffsynth.extensions.ImageQualityMetric.clip import CLIPScore
from diffsynth.extensions.ImageQualityMetric.fvd import compute_fvd, compute_fvd_v

class ReCamMasterEvaluator:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = torch.device(device)
        self.clip_scorer = CLIPScore(self.device)
           
    def extract_all_frames_for_fid(self, video_path: str, output_dir: str, video_id: int, max_frames: int = 60, size: Tuple[int, int] = (299, 299)):
        """为FID计算提取所有帧（优化版本）"""
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        # 获取视频总帧数，进行均匀采样
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # 均匀采样帧
        if total_frames <= max_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_pil = frame_pil.resize(size, Image.BICUBIC)
            
            # 保存帧（使用JPEG格式减少I/O时间）
            frame_path = os.path.join(output_dir, f"video_{video_id}_frame_{frame_count:06d}.jpg")
            frame_pil.save(frame_path, "JPEG", quality=85)
            frame_count += 1
        
        cap.release()
        return frame_count
    
    def calculate_fid(self, real_dir: str, fake_dir: str) -> float:
        """计算FID (Frechet Inception Distance)"""
        try:
            from pytorch_fid import fid_score
            
            # 计算FID
            fid_value = fid_score.calculate_fid_given_paths(
                [real_dir, fake_dir],
                batch_size=16,
                device=self.device.type,
                dims=2048,
                num_workers=4  # 减少worker数量
            )
            return fid_value
        except Exception as e:
            print(f"FID计算错误: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
    

 
    def extract_video_frames(self, video_path: str, num_frames: int = 60) -> List[Image.Image]:
        """从视频中提取帧"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            print(f"警告: 视频 {video_path} 帧数为0")
            cap.release()
            return frames
        
        # 均匀采样帧
        frame_indices = np.linspace(0, total_frames - 1, min(num_frames, total_frames), dtype=int)
        
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if i in frame_indices:
                # 转换BGR到RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
        
        cap.release()
        return frames
    
    def videos_to_tensors(self, video_paths: List[str], num_frames: int = 60) -> torch.Tensor:
        """将视频转换为tensor格式用于FVD计算"""
        video_tensors = []
        
        for video_path in tqdm(video_paths, desc="处理视频"):
            try:
                # 提取视频帧
                frames = self.extract_video_frames(video_path, num_frames=num_frames)
                if not frames:
                    continue
                
                # 将PIL图像转换为tensor
                frame_tensors = []
                for frame in frames:
                    # 转换为RGB并resize到224x224（I3D模型的标准输入尺寸）
                    frame_rgb = frame.convert('RGB').resize((224, 224))
                    frame_tensor = torch.from_numpy(np.array(frame_rgb)).float() / 255.0
                    frame_tensor = frame_tensor.permute(2, 0, 1)  # HWC -> CHW
                    frame_tensors.append(frame_tensor)
                
                # 堆叠帧：[T, C, H, W]
                video_tensor = torch.stack(frame_tensors, dim=0)
                # 转换为 [C, T, H, W] 格式
                video_tensor = video_tensor.permute(1, 0, 2, 3)
                video_tensors.append(video_tensor)
                
            except Exception as e:
                print(f"处理视频失败 {video_path}: {e}")
                continue
        
        if not video_tensors:
            return torch.empty(0)
        
        # 堆叠所有视频tensor
        return torch.stack(video_tensors, dim=0)  # [N, C, T, H, W]
    
    def calculate_fvd(self, real_videos: List[str], fake_videos: List[str]) -> float:
        """计算FVD (Frechet Video Distance)"""
        try:
            # 提取视频帧并转换为tensor
            real_tensors = self.videos_to_tensors(real_videos)
            fake_tensors = self.videos_to_tensors(fake_videos)
            
            if len(real_tensors) == 0 or len(fake_tensors) == 0:
                print("没有有效的视频数据用于FVD计算")
                return 0.0
            
            # 确保tensor维度正确
            if real_tensors.dim() != 5 or fake_tensors.dim() != 5:
                print(f"视频tensor维度错误: 真实视频 {real_tensors.dim()}, 生成视频 {fake_tensors.dim()}")
                return 0.0
                
            # 计算FVD
            fvd_value = compute_fvd(
                y_true=real_tensors,
                y_pred=fake_tensors,
                max_items=min(len(real_tensors), len(fake_tensors)),
                device=self.device,
                batch_size=4
            )
            return fvd_value
            
        except Exception as e:
            print(f"FVD计算错误: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
    

    def prepare_fvd_v_data(self, real_tensors: torch.Tensor, fake_tensors: torch.Tensor, num_views: int = 1):
        """为FVD-V准备数据格式"""
        # 对于单视图视频，我们使用相同的数据复制到视图维度
        # 在实际应用中，你应该加载真实的多视图视频数据
        
        # real_tensors, fake_tensors 形状: [N, C, T, H, W]
        
        # 添加视图维度: [N, C, T, H, W] -> [N, T, V, C, H, W]
        real_views = real_tensors.permute(0, 2, 1, 3, 4).unsqueeze(2)  # 在位置2添加视图维度
        fake_views = fake_tensors.permute(0, 2, 1, 3, 4).unsqueeze(2)  # 在位置2添加视图维度
        
        # 如果num_views > 1，复制数据到多个视图（仅用于演示）
        if num_views > 1:
            real_views = real_views.repeat(1, 1, num_views, 1, 1, 1)
            fake_views = fake_views.repeat(1, 1, num_views, 1, 1, 1)
        
        return real_views, fake_views

    def calculate_fvd_v(self, real_videos: List[str], fake_videos: List[str], num_views: int = 1) -> float:
        """计算FVD-V：每个时间帧上不同视图之间的FVD"""
        try:
            # 提取视频帧并转换为tensor
            real_tensors = self.videos_to_tensors(real_videos)
            fake_tensors = self.videos_to_tensors(fake_videos)
            
            if len(real_tensors) == 0 or len(fake_tensors) == 0:
                print("没有有效的视频数据用于FVD-V计算")
                return 0.0
            
            # 准备FVD-V数据格式
            real_views, fake_views = self.prepare_fvd_v_data(real_tensors, fake_tensors, num_views)
            
            if real_views.dim() != 6 or fake_views.dim() != 6:
                print(f"FVD-V数据维度错误: 真实视频 {real_views.dim()}, 生成视频 {fake_views.dim()}")
                return 0.0
                
            # 计算FVD-V
            fvd_v_value = compute_fvd_v(
                y_true=real_views,
                y_pred=fake_views,
                max_items=min(len(real_views), len(fake_views)),
                device=self.device,
                batch_size=4
            )
            return fvd_v_value
            
        except Exception as e:
            print(f"FVD-V计算错误: {e}")
            import traceback
            traceback.print_exc()
            return 0.0


    def calculate_clip_t(self, frames: List[Image.Image], prompt: str) -> float:
        """计算CLIP-T：每帧与文本提示的相似度"""
        scores = self.clip_scorer.score(frames, prompt)
        return np.mean(scores)
    
    def calculate_clip_f(self, frames: List[Image.Image]) -> float:
        """计算CLIP-F：相邻帧之间的相似度"""
        if len(frames) < 2:
            return 0.0
        
        # 提取所有帧的特征
        frame_features = []
        for frame in frames:
            with torch.no_grad():
                # 使用CLIP模型提取图像特征
                image_tensor = self.clip_scorer.preprocess_val(frame).unsqueeze(0).to(self.device)
                features = self.clip_scorer.model.encode_image(image_tensor)
                features = F.normalize(features, p=2, dim=1)
                frame_features.append(features.cpu())
        
        # 计算相邻帧的相似度
        similarities = []
        for i in range(len(frame_features) - 1):
            sim = F.cosine_similarity(frame_features[i], frame_features[i+1])
            similarities.append(sim.item())
        
        return np.mean(similarities)
    
    def calculate_clip_v(self, source_frames: List[Image.Image], target_frames: List[Image.Image]) -> float:
        """计算CLIP-V：源视频和目标视频对应帧的相似度"""
        min_frames = min(len(source_frames), len(target_frames))
        if min_frames == 0:
            return 0.0
        
        # 对齐帧数
        source_frames = source_frames[:min_frames]
        target_frames = target_frames[:min_frames]
        
        # 提取特征并计算相似度
        similarities = []
        for src_frame, tgt_frame in zip(source_frames, target_frames):
            with torch.no_grad():
                # 处理源帧
                src_tensor = self.clip_scorer.preprocess_val(src_frame).unsqueeze(0).to(self.device)
                src_features = self.clip_scorer.model.encode_image(src_tensor)
                src_features = F.normalize(src_features, p=2, dim=1)
                
                # 处理目标帧
                tgt_tensor = self.clip_scorer.preprocess_val(tgt_frame).unsqueeze(0).to(self.device)
                tgt_features = self.clip_scorer.model.encode_image(tgt_tensor)
                tgt_features = F.normalize(tgt_features, p=2, dim=1)
                
                # 计算相似度
                sim = F.cosine_similarity(src_features, tgt_features)
                similarities.append(sim.item())
        
        return np.mean(similarities)
    
    def evaluate_single_pair(self, source_video_path: str, target_video_path: str, prompt: str) -> Dict:
        """评估单个源视频-目标视频对"""
        # 提取帧
        source_frames = self.extract_video_frames(source_video_path)
        target_frames = self.extract_video_frames(target_video_path)
        
        # 计算各项指标
        clip_t = self.calculate_clip_t(target_frames, prompt)
        clip_f = self.calculate_clip_f(target_frames)
        clip_v = self.calculate_clip_v(source_frames, target_frames)
        # clip_t = 0.0
        # clip_f = 0.0
        # clip_v = 0.0
        
        return {
            'clip_t': clip_t,
            'clip_f': clip_f,
            'clip_v': clip_v,
            'source_video': os.path.basename(source_video_path),
            'target_video': os.path.basename(target_video_path)
        }
    
    def evaluate_dataset_level_metrics(self, source_videos_dir: str, target_videos_dir: str, metadata_path: str, output_dir: str = "metrics_output", max_videos: int = 100):
        """计算数据集级别的指标 (FID, FVD, FVD-V)"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 读取metadata
        metadata_df = pd.read_csv(metadata_path)
        filename_to_prompt = {}
        for _, row in metadata_df.iterrows():
            filename_to_prompt[row['file_name']] = row['text']
        
        # 获取视频文件列表
        source_videos_path = Path(source_videos_dir)
        target_videos_path = Path(target_videos_dir)
        
        source_video_files = list(source_videos_path.glob("*.mp4"))
        target_video_files = list(target_videos_path.glob("*.mp4"))
        
        # 只处理有对应关系的视频
        matched_source_videos = []
        matched_target_videos = []
        
        for target_video in target_video_files:
            if target_video.name in filename_to_prompt:
                source_video = source_videos_path / target_video.name
                if source_video.exists():
                    matched_source_videos.append(str(source_video))
                    matched_target_videos.append(str(target_video))
        
        print(f"找到 {len(matched_source_videos)} 对匹配的视频用于数据集级别评估")
        
        metrics = {}
        
        # FID
        max_videos_for_fid = min(max_videos, len(matched_source_videos))
        print(f"FID计算将处理 {max_videos_for_fid} 个视频对")
        with tempfile.TemporaryDirectory() as real_dir, tempfile.TemporaryDirectory() as fake_dir:
            # 提取真实和生成视频的帧
            for i in range(max_videos_for_fid):
                source_video = matched_source_videos[i]
                target_video = matched_target_videos[i]
                
                print(f"处理视频对 {i+1}/{max_videos_for_fid}: {os.path.basename(source_video)}")
                
                real_frames = self.extract_all_frames_for_fid(source_video, real_dir, i)
                fake_frames = self.extract_all_frames_for_fid(target_video, fake_dir, i)
            
        
            print("特征提取完毕，正在计算FID...")
            fid_value = self.calculate_fid(real_dir, fake_dir)
            metrics['fid'] = fid_value
            print(f"FID: {fid_value:.2f}")
        
        # FVD
        print(f"FVD计算将处理 {max_videos} 个视频对")
        source_subset = matched_source_videos[:max_videos]
        target_subset = matched_target_videos[:max_videos]
        fvd_value = self.calculate_fvd(source_subset, target_subset)
        metrics['fvd'] = fvd_value
        print(f"FVD: {fvd_value:.2f}")

        # # 呃等你做完了至少3个视角再来说这个FVD的事情吧
        # # FVD-V
        # print(f"FVD-V计算将处理 {max_videos} 个视频对")
        # fvd_v_value = self.calculate_fvd_v(source_subset, target_subset, num_views=1)
        # metrics['fvd_v'] = fvd_v_value
        # print(f"FVD-V: {fvd_v_value:.2f}")
            
        
        # 保存数据集级别指标
        metrics_path = os.path.join(output_dir, "dataset_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
    
    def evaluate_all(self, source_videos_dir: str, target_videos_dir: str, metadata_path: str, 
                    output_path: str = "evaluation_results.json", include_dataset_metrics: bool = True,
                    max_videos_for_metrics: int = 50):
        """评估所有视频对"""
        # 读取metadata
        metadata_df = pd.read_csv(metadata_path)
        
        # 创建文件名到文本的映射
        filename_to_prompt = {}
        for _, row in metadata_df.iterrows():
            filename_to_prompt[row['file_name']] = row['text']
        
        # 获取目标视频文件列表
        target_videos_path = Path(target_videos_dir)
        target_video_files = list(target_videos_path.glob("*.mp4"))
        
        results = []
        print(f"开始评估 {len(target_video_files)} 个视频...")
        
        for target_video_file in tqdm(target_video_files):
            target_filename = target_video_file.name
            
            # 查找对应的源视频
            if target_filename in filename_to_prompt:
                prompt = filename_to_prompt[target_filename]
                source_video_path = Path(source_videos_dir) / target_filename
                
                if source_video_path.exists():
                    try:
                        result = self.evaluate_single_pair(
                            str(source_video_path),
                            str(target_video_file),
                            prompt
                        )
                        results.append(result)
                    except Exception as e:
                        print(f"评估 {target_filename} 时出错: {e}")
                else:
                    print(f"源视频不存在: {source_video_path}")
            else:
                print(f"未找到 {target_filename} 的文本提示")
        
        # 计算平均指标
        if results:
            avg_metrics = {
                'avg_clip_t': np.mean([r['clip_t'] for r in results]),
                'avg_clip_f': np.mean([r['clip_f'] for r in results]),
                'avg_clip_v': np.mean([r['clip_v'] for r in results]),
                'total_videos': len(results)
            }
            
            # # 计算数据集级别指标
            # if include_dataset_metrics:
            #     print(f"\n计算数据集级别指标（最多处理 {max_videos_for_metrics} 个视频对）...")
            #     dataset_metrics = self.evaluate_dataset_level_metrics(
            #         source_videos_dir, target_videos_dir, metadata_path, max_videos=max_videos_for_metrics
            #     )
            #     avg_metrics.update(dataset_metrics)
            
            # 保存结果
            output_data = {
                'individual_results': results,
                'average_metrics': avg_metrics
            }
            
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            print(f"\n评估完成!")
            print(f"评估了 {len(results)} 个视频对")
            print(f"平均 CLIP-T: {avg_metrics['avg_clip_t']:.6f}")
            print(f"平均 CLIP-F: {avg_metrics['avg_clip_f']:.6f}")
            print(f"平均 CLIP-V: {avg_metrics['avg_clip_v']:.6f}")
            
            # if 'fid' in avg_metrics:
            #     print(f"FID: {avg_metrics['fid']:.6f}")
            # if 'fvd' in avg_metrics:
            #     print(f"FVD: {avg_metrics['fvd']:.6f}")
            # if 'fvd_v' in avg_metrics:
            #     print(f"FVD-V: {avg_metrics['fvd_v']:.6f}")
                
            print(f"详细结果已保存到: {output_path}")
        
        return results



if __name__ == "__main__":
    # 配置路径
    source_videos_dir = "/data/wlh/ReCamMaster/WebVID/videos"  # 原始WebVID视频
    target_videos_dir = "/data/wlh/ReCamMaster/ReCamMaster-main/results/cam_type1"  # ReCamMaster生成的视频
    metadata_path = "/data/wlh/ReCamMaster/WebVID/metadata-all.csv"  # 之前生成的metadata文件
    output_path = "/data/wlh/ReCamMaster/ReCamMaster-main/evaluation_results.json"
    
    # 初始化评估器
    evaluator = ReCamMasterEvaluator()
    
    # 执行评估
    results = evaluator.evaluate_all(
        source_videos_dir=source_videos_dir,
        target_videos_dir=target_videos_dir,
        metadata_path=metadata_path,
        output_path=output_path,
        include_dataset_metrics=True,  # 设置为False可以跳过FID/FVD计算
        max_videos_for_metrics=300  # 限制处理的视频数量以避免卡死
    )