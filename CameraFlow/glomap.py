# conda activate wan
# CUDA_VISIBLE_DEVICES=2 python glomap-cam1.py

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
import struct
from scipy.spatial.transform import Rotation as R


import os
import subprocess

def run_glomap_simple(image_file: str, timeout: int = 120) -> bool:
    """
    执行Glomap的三个基本步骤（带超时机制）
    
    参数:
        image_file: 图片存放的文件夹路径
        timeout: 每个步骤的超时时间（秒）
    """
    
    # 确保输出目录存在
    glomap_output_dir = f"{image_file}/glomap"
    os.makedirs(glomap_output_dir, exist_ok=True)
    
    try:
        # 1. colmap feature_extractor
        feature_extractor_cmd = [
            "colmap", "feature_extractor",
            "--image_path", image_file,
            "--database_path", f"{image_file}/database.db"
        ]
        
        print(f"  执行特征提取: {image_file}")
        result1 = subprocess.run(
            feature_extractor_cmd, 
            capture_output=True, 
            text=True,
            timeout=timeout
        )
        if result1.returncode != 0:
            print(f"  特征提取失败: {result1.stderr}")
            return False
        
        # 2. colmap exhaustive_matcher
        matcher_cmd = [
            "colmap", "exhaustive_matcher",
            "--database_path", f"{image_file}/database.db"
        ]
        
        print(f"  执行特征匹配: {image_file}")
        result2 = subprocess.run(
            matcher_cmd, 
            capture_output=True, 
            text=True,
            timeout=timeout
        )
        if result2.returncode != 0:
            print(f"  特征匹配失败: {result2.stderr}")
            return False
        
        # 3. glomap mapper
        mapper_cmd = [
            "glomap", "mapper",
            "--database_path", f"{image_file}/database.db",
            "--image_path", image_file,
            "--output_path", glomap_output_dir
        ]
        
        print(f"  执行Glomap映射: {image_file}")
        result3 = subprocess.run(
            mapper_cmd, 
            capture_output=True, 
            text=True,
            timeout=timeout
        )
        if result3.returncode != 0:
            print(f"  Glomap映射失败: {result3.stderr}")
            return False
        
        return True
        
    except subprocess.TimeoutExpired:
        print(f"  GLOMAP处理超时（超过{timeout}秒），跳过此视频")
        return False
    except Exception as e:
        print(f"  GLOMAP处理异常: {e}")
        return False

def parse_matrix_string(matrix_str: str) -> np.ndarray:
    """解析矩阵字符串为numpy数组"""
    # 移除括号和空格，然后分割
    matrix_str = matrix_str.replace('[', '').replace(']', '').strip()
    rows = matrix_str.split('] [')
    matrix = []
    for row in rows:
        elements = row.split()
        matrix.append([float(x) for x in elements])
    return np.array(matrix)

def read_cameras_bin(bin_file_path: str) -> Dict:
    """读取cameras.bin二进制文件"""
    cameras = {}
    with open(bin_file_path, 'rb') as f:
        num_cameras = struct.unpack('Q', f.read(8))[0]
        for _ in range(num_cameras):
            camera_id = struct.unpack('I', f.read(4))[0]
            model_id = struct.unpack('I', f.read(4))[0]
            width = struct.unpack('Q', f.read(8))[0]
            height = struct.unpack('Q', f.read(8))[0]
            
            # 读取参数
            num_params = struct.unpack('Q', f.read(8))[0]
            params = []
            for __ in range(num_params):
                params.append(struct.unpack('d', f.read(8))[0])
            
            cameras[camera_id] = {
                'model_id': model_id,
                'width': width,
                'height': height,
                'params': params
            }
    return cameras

def read_images_bin(bin_file_path: str) -> Dict:
    """读取images.bin二进制文件，获取相机位姿"""
    images = {}
    with open(bin_file_path, 'rb') as f:
        num_images = struct.unpack('Q', f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack('I', f.read(4))[0]
            
            # 读取四元数 (qw, qx, qy, qz)
            qw = struct.unpack('d', f.read(8))[0]
            qx = struct.unpack('d', f.read(8))[0]
            qy = struct.unpack('d', f.read(8))[0]
            qz = struct.unpack('d', f.read(8))[0]
            
            # 读取平移向量
            tx = struct.unpack('d', f.read(8))[0]
            ty = struct.unpack('d', f.read(8))[0]
            tz = struct.unpack('d', f.read(8))[0]
            
            camera_id = struct.unpack('I', f.read(4))[0]
            
            # 读取图像名称
            name_length = 0
            name_chars = []
            while True:
                char = f.read(1)
                if char == b'\x00':
                    break
                name_chars.append(char)
                name_length += 1
            image_name = b''.join(name_chars).decode('utf-8')
            
            # 跳过2D点数据
            num_points2D = struct.unpack('Q', f.read(8))[0]
            for __ in range(num_points2D):
                f.read(8 * 2)  # x, y
                f.read(8)      # point3D_id
            
            # 将四元数转换为旋转矩阵
            rotation = R.from_quat([qx, qy, qz, qw]).as_matrix()
            translation = np.array([tx, ty, tz])
            
            images[image_name] = {
                'rotation': rotation,
                'translation': translation,
                'camera_id': camera_id
            }
    return images

def calculate_rotation_error(R1: np.ndarray, R2: np.ndarray) -> float:
    """计算旋转误差 (RotErr)"""
    R_rel = R1 @ R2.T
    trace = np.trace(R_rel)
    cos_theta = np.clip((trace - 1) / 2, -1.0, 1.0)
    return np.arccos(cos_theta)

def calculate_translation_error(t1: np.ndarray, t2: np.ndarray) -> float:
    """计算平移误差 (TransErr)"""
    return np.linalg.norm(t1 - t2)

def evaluate_camera_accuracy(glomap_output_dir: str, gt_extrinsics_path: str) -> Tuple[float, float]:
    """
    评估相机位姿准确性
    
    参数:
        glomap_output_dir: GLOMAP输出目录
        gt_extrinsics_path: 真实外参文件路径
        
    返回:
        avg_rot_err: 平均旋转误差
        avg_trans_err: 平均平移误差
    """
    
    # 读取真实外参
    with open(gt_extrinsics_path, 'r') as f:
        gt_extrinsics = json.load(f)
    
    # 读取GLOMAP估计的位姿
    sparse_dir = f"{glomap_output_dir}/0"
    images_bin_path = f"{sparse_dir}/images.bin"
    
    if not os.path.exists(images_bin_path):
        print(f"警告: {images_bin_path} 不存在")
        return float('inf'), float('inf')
    
    estimated_poses = read_images_bin(images_bin_path)
    
    rot_errors = []
    trans_errors = []
    
    # 对每一帧计算误差
    for frame_name, estimated_pose in estimated_poses.items():
        # 从文件名提取帧索引 (例如: "frame_000000.jpg" -> "0")
        frame_idx = int(frame_name.replace('frame_', '').replace('.jpg', ''))
        frame_key = f"frame{frame_idx}"
        
        if frame_key in gt_extrinsics and "cam02" in gt_extrinsics[frame_key]:
            # 获取真实外参
            gt_matrix_str = gt_extrinsics[frame_key]["cam02"]
            gt_matrix = parse_matrix_string(gt_matrix_str)
            
            # 提取真实旋转和平移
            gt_rotation = gt_matrix[:3, :3]
            gt_translation = gt_matrix[:3, 3]
            
            # 获取估计的旋转和平移
            est_rotation = estimated_pose['rotation']
            est_translation = estimated_pose['translation']
            
            # 计算误差
            rot_err = calculate_rotation_error(est_rotation, gt_rotation)
            trans_err = calculate_translation_error(est_translation, gt_translation)
            
            rot_errors.append(rot_err)
            trans_errors.append(trans_err)
    
    if not rot_errors:
        print("警告: 没有找到匹配的帧进行评估")
        return float('inf'), float('inf')
    
    avg_rot_err = np.mean(rot_errors)
    avg_trans_err = np.mean(trans_errors)
    
    # print(f"相机位姿评估结果:")
    # print(f"  平均旋转误差 (RotErr): {avg_rot_err:.6f} 弧度")
    # print(f"  平均平移误差 (TransErr): {avg_trans_err:.6f}")
    # print(f"  评估帧数: {len(rot_errors)}")
    
    return avg_rot_err, avg_trans_err

def extract_frames(videos_dir, num_frames, gt_extrinsics_path, output_csv_path):
    # 获取视频文件列表
    videos_path = Path(videos_dir)
    video_files = list(videos_path.glob("*.mp4"))
    
    all_results = []
    processed_count = 0
    failed_count = 0
    
    # 准备CSV文件
    csv_results = []
    
    for video_file in tqdm(video_files, desc="Extracting, GLOMAPing, Analyzing Frames..."):
        processed_count += 1
        print(f"\n处理视频 {processed_count}/{len(video_files)}: {video_file.stem}")
        
        try:
            # 提取帧到特定文件夹
            tmp_dir = os.path.join(videos_dir, video_file.stem)
            os.makedirs(tmp_dir, exist_ok=True)
            
            # # 如果目录已存在，则直接执行就好了，你不存在才回去乱搞创建
            # if not os.path.exists(tmp_dir):
            #     cap = cv2.VideoCapture(str(video_file))
            #     # 获取视频总帧数，进行均匀采样
            #     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
            #     # 如果视频帧数不足，跳过
            #     if total_frames < 10:
            #         print(f"  视频帧数不足({total_frames}帧)，跳过")
            #         result = {
            #             'video': video_file.stem,
            #             'rot_err': float('inf'),
            #             'trans_err': float('inf'),
            #             'status': 'insufficient_frames'
            #         }
            #         all_results.append(result)
            #         csv_results.append([video_file.stem, float('inf'), float('inf')])
            #         cap.release()
            #         continue
                    
            #     # 均匀采样帧
            #     if total_frames <= num_frames:
            #         frame_indices = list(range(total_frames))
            #     else:
            #         frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
                
            #     frame_count = 0
            #     for i in frame_indices:
            #         cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            #         ret, frame = cap.read()
            #         if not ret:
            #             continue
            #         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #         frame_pil = Image.fromarray(frame_rgb)
            #         # 保存帧（使用JPEG格式减少I/O时间）
            #         frame_path = os.path.join(tmp_dir, f"frame_{frame_count:06d}.jpg")
            #         frame_pil.save(frame_path, "JPEG", quality=85)
            #         frame_count += 1
            #     cap.release()


            #     # 执行 glomap 估计（带超时）
            #     success = run_glomap_simple(tmp_dir, timeout=80)  # 1分钟超时
            
            #     if not success:
            #         print(f"  GLOMAP处理失败，跳过此视频")
            #         failed_count += 1
            #         result = {
            #             'video': video_file.stem,
            #             'rot_err': float('inf'),
            #             'trans_err': float('inf'),
            #             'status': 'glomap_failed'
            #         }
            #         all_results.append(result)
            #         csv_results.append([video_file.stem, float('inf'), float('inf')])
            #         continue

            # 执行 Camera Accuracy 评估
            print(f"  开始相机精度评估...")
            glomap_output_dir = f"{tmp_dir}/glomap"
            rot_err, trans_err = evaluate_camera_accuracy(glomap_output_dir, gt_extrinsics_path)
            
            result = {
                'video': video_file.stem,
                'rot_err': float(rot_err),
                'trans_err': float(trans_err),
                'status': 'success'
            }
            all_results.append(result)
            csv_results.append([video_file.stem, float(rot_err), float(trans_err)])
            
        except Exception as e:
            print(f"  处理视频 {video_file.stem} 时发生异常: {e}")
            failed_count += 1
            result = {
                'video': video_file.stem,
                'rot_err': float('inf'),
                'trans_err': float('inf'),
                'status': 'exception'
            }
            all_results.append(result)
            csv_results.append([video_file.stem, float('inf'), float('inf')])
            continue
    
    # 保存结果到CSV文件
    df = pd.DataFrame(csv_results, columns=['filename', 'RotErr', 'TransErr'])
    df.to_csv(output_csv_path, index=False)
    print(f"\n结果已保存到: {output_csv_path}")
    
    # 输出总体结果
    print("\n" + "="*60)
    print("总体评估结果:")
    print("="*60)
    
    success_results = [r for r in all_results if r['status'] == 'success']
    failed_results = [r for r in all_results if r['status'] != 'success']
    
    for result in all_results:
        status_icon = "✅" if result['status'] == 'success' else "❌"
        print(f"{status_icon} {result['video']}: RotErr={result['rot_err']:.6f}, TransErr={result['trans_err']:.6f} [{result['status']}]")
    
    if success_results:
        avg_rot_err = np.mean([r['rot_err'] for r in success_results])
        avg_trans_err = np.mean([r['trans_err'] for r in success_results])
        print(f"\n成功处理: {len(success_results)}/{len(all_results)} 个视频")
        print(f"平均值: RotErr={avg_rot_err:.6f}, TransErr={avg_trans_err:.6f}")
    else:
        print(f"\n没有成功处理的视频")
    
    if failed_results:
        print(f"失败详情:")
        for fail in failed_results:
            print(f"  - {fail['video']}: {fail['status']}")
    
    return all_results

if __name__ == "__main__":
    # 配置路径
    videos_dir = "/data/wlh/ReCamMaster/ReCamMaster-main/results/cam_type2"  # ReCamMaster生成的视频
    gt_extrinsics_path = "/data/wlh/ReCamMaster/ReCamMaster-main/example_test_data/cameras/camera_extrinsics.json"  # 请替换为实际路径
    output_csv_path = "/data/wlh/ReCamMaster/ReCamMaster-main/glomap_results.csv"  # 输出CSV文件路径
    
    extract_frames(
        videos_dir=videos_dir,
        num_frames=100,
        gt_extrinsics_path=gt_extrinsics_path,
        output_csv_path=output_csv_path
    )