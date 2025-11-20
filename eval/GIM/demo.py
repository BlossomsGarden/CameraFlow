# -*- coding: utf-8 -*-
# conda env  create -f environment.yaml
# ransac_reproj_threshold 取默认值 8.0
# 模型采样  model.sample() 采样11000个点
# CUDA_VISIBLE_DEVICES=5 python demo.py  --model gim_dkm   --recam_folder /data/wlh/ReCamMaster/ReCamMaster-main/results/cam_type1/     --original_folder /data/wlh/ReCamMaster/WebVID/videos/     --num_frames 80
#
# 如果报错xformers cuda问题，直接重装 conda install xformers -c xformers


import cv2
import torch
import argparse
import warnings
import numpy as np
import os
import glob
from os.path import join, basename, exists
from tools import get_padding_size
from networks.roma.roma import RoMa
from networks.dkm.models.model_zoo.DKMv3 import DKMv3

# Constants
DEFAULT_RANSAC_MAX_ITER = 10000
DEFAULT_RANSAC_CONFIDENCE = 0.999
DEFAULT_RANSAC_REPROJ_THRESHOLD = 8


def preprocess(image: np.ndarray, grayscale: bool = False, resize_max: int = None, dfactor: int = 8):
    image = image.astype(np.float32, copy=False)
    size = image.shape[:2][::-1]
    scale = np.array([1.0, 1.0])

    if grayscale:
        assert image.ndim == 2, image.shape
        image = image[None]
    else:
        image = image.transpose((2, 0, 1))
    image = torch.from_numpy(image / 255.0).float()

    size_new = tuple(map(lambda x: int(x // dfactor * dfactor), image.shape[-2:]))
    image = torch.nn.functional.interpolate(image[None], size=size_new, mode='bilinear', align_corners=False)[0]
    scale = np.array(size) / np.array(size_new)[::-1]
    return image, scale

def extract_frames(video_path, num_frames=60):
    """从视频中提取指定数量的帧"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: Cannot open video {video_path}")
        return []
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < num_frames:
        num_frames = total_frames
    
    if num_frames == 0:
        cap.release()
        return []
    
    frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    frames = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    
    cap.release()
    return frames

def count_matching_pixels(kpts0, kpts1, ransac_reproj_threshold=8.0):
    """计算匹配像素点数量"""
    if len(kpts0) < 8:
        return 0
    
    kpts0_np = kpts0.cpu().detach().numpy() if torch.is_tensor(kpts0) else kpts0
    kpts1_np = kpts1.cpu().detach().numpy() if torch.is_tensor(kpts1) else kpts1
    
    _, mask = cv2.findFundamentalMat(
        kpts0_np,
        kpts1_np,
        method=cv2.USAC_MAGSAC,
        ransacReprojThreshold=ransac_reproj_threshold,
        confidence=0.999999,
        maxIters=10000
    )
    
    if mask is not None:
        return np.sum(mask.ravel() > 0)
    return 0

def load_model(model_name, device):
    """加载指定的匹配模型"""
    model = None
    detector = None
    
    if model_name == 'gim_dkm':
        model = DKMv3(weights=None, h=672, w=896)
        ckpt_path = 'weights/gim_dkm_100h.ckpt'
    elif model_name == 'gim_roma':
        model = RoMa(img_size=[672])
        ckpt_path = 'weights/gim_roma_100h.ckpt'
    
    # 检查权重文件是否存在
    if not exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
    
    # 加载权重
    state_dict = torch.load(ckpt_path, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    
    # 处理权重加载
    for k in list(state_dict.keys()):
        if k.startswith('model.'):
            state_dict[k.replace('model.', '', 1)] = state_dict.pop(k)
        if model_name == 'gim_dkm' and 'encoder.net.fc' in k:
            state_dict.pop(k)
    model.load_state_dict(state_dict)
    
    model = model.eval().to(device)
    return model, detector

def match_images(model, detector, image0, image1, device, model_name):
    """对两幅图像进行匹配"""
    image0_proc, scale0 = preprocess(image0)
    image1_proc, scale1 = preprocess(image1)
    
    image0_proc = image0_proc.to(device)[None]
    image1_proc = image1_proc.to(device)[None]
    
    data = dict(color0=image0_proc, color1=image1_proc, image0=image0_proc, image1=image1_proc)
    
    try:
        if model_name in ['gim_dkm', 'gim_roma']:
            width, height = (672, 896) if model_name == 'gim_dkm' else (672, 672)
            
            orig_width0, orig_height0, pad_left0, pad_right0, pad_top0, pad_bottom0 = get_padding_size(image0_proc, width, height)
            orig_width1, orig_height1, pad_left1, pad_right1, pad_top1, pad_bottom1 = get_padding_size(image1_proc, width, height)
            
            image0_padded = torch.nn.functional.pad(image0_proc, (pad_left0, pad_right0, pad_top0, pad_bottom0))
            image1_padded = torch.nn.functional.pad(image1_proc, (pad_left1, pad_right1, pad_top1, pad_bottom1))
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dense_matches, dense_certainty = model.match(image0_padded, image1_padded)
                sparse_matches, mconf = model.sample(dense_matches, dense_certainty, 11000)
            
            height0, width0 = image0_padded.shape[-2:]
            height1, width1 = image1_padded.shape[-2:]
            
            kpts0 = sparse_matches[:, :2]
            kpts0 = torch.stack((
                width0 * (kpts0[:, 0] + 1) / 2, height0 * (kpts0[:, 1] + 1) / 2), dim=-1)
            kpts1 = sparse_matches[:, 2:]
            kpts1 = torch.stack((
                width1 * (kpts1[:, 0] + 1) / 2, height1 * (kpts1[:, 1] + 1) / 2), dim=-1)
            
            # 移除padding的影响
            kpts0 -= kpts0.new_tensor((pad_left0, pad_top0))[None]
            kpts1 -= kpts1.new_tensor((pad_left1, pad_top1))[None]
            
            mask_ = (kpts0[:, 0] > 0) & (kpts0[:, 1] > 0) & (kpts1[:, 0] > 0) & (kpts1[:, 1] > 0)
            mask_ = mask_ & (kpts0[:, 0] <= (orig_width0 - 1)) & (kpts1[:, 0] <= (orig_width1 - 1)) & \
                    (kpts0[:, 1] <= (orig_height0 - 1)) & (kpts1[:, 1] <= (orig_height1 - 1))
            
            mconf = mconf[mask_]
            kpts0 = kpts0[mask_]
            kpts1 = kpts1[mask_]
            
        return kpts0, kpts1, mconf
        
    except Exception as e:
        print(f"Error during matching: {e}")
        return None, None, None

def find_video_pairs(recam_folder, original_folder):
    """在ReCamMaster文件夹和原始视频文件夹中寻找对应的视频对"""
    recam_videos = glob.glob(join(recam_folder, "*.mp4"))
    video_pairs = []
    
    for recam_video in recam_videos:
        video_name = basename(recam_video)
        original_video = join(original_folder, video_name)
        
        if exists(original_video):
            video_pairs.append((recam_video, original_video, video_name))
        else:
            print(f"Warning: Original video not found for {video_name}")
    
    return video_pairs

def evaluate_video_pair(model, detector, recam_video, original_video, num_frames, device, model_name):
    """评估一对视频的匹配像素点"""
    print(f"Processing: {basename(recam_video)}")
    
    # 提取帧
    recam_frames = extract_frames(recam_video, num_frames)
    original_frames = extract_frames(original_video, num_frames)
    
    if len(recam_frames) == 0 or len(original_frames) == 0:
        print(f"Warning: Could not extract frames from one of the videos")
        return 0.0
    
    num_pairs = min(len(recam_frames), len(original_frames))
    print(f"  Extracted {num_pairs} frame pairs")
    
    # 计算匹配像素点
    total_matching_pixels = 0
    valid_pairs = 0
    
    for i in range(num_pairs):
        kpts0, kpts1, mconf = match_images(model, detector, recam_frames[i], original_frames[i], device, model_name)
        
        if kpts0 is not None and len(kpts0) > 0:
            matching_pixels = count_matching_pixels(kpts0, kpts1)
            total_matching_pixels += matching_pixels
            valid_pairs += 1
    
    if valid_pairs == 0:
        return 0.0
    
    # 计算平均值（以千为单位）
    avg_matching_pixels_k = total_matching_pixels / 1000
    
    print(f"  Average Matching Pixels: {avg_matching_pixels_k:.3f}k")
    print(f"  Valid frame pairs: {valid_pairs}/{num_pairs}")
    
    return avg_matching_pixels_k

def main():
    parser = argparse.ArgumentParser(description='Calculate matching pixels between ReCamMaster and original videos')
    parser.add_argument('--model', type=str, default='gim_dkm', choices=['gim_roma', 'gim_dkm'])
    parser.add_argument('--recam_folder', type=str, required=True, help='Path to folder containing ReCamMaster generated videos')
    parser.add_argument('--original_folder', type=str, required=True, help='Path to folder containing original WebVID videos')
    parser.add_argument('--num_frames', type=int, default=80, help='Number of frames to extract from each video')
    
    args = parser.parse_args()
    
    # 检查文件夹是否存在
    if not exists(args.recam_folder):
        raise FileNotFoundError(f"ReCamMaster folder not found: {args.recam_folder}")
    if not exists(args.original_folder):
        raise FileNotFoundError(f"Original videos folder not found: {args.original_folder}")
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Model: {args.model}")
    print(f"ReCamMaster folder: {args.recam_folder}")
    print(f"Original videos folder: {args.original_folder}")
    print(f"Frames per video: {args.num_frames}")
    print("-" * 50)
    
    # 加载模型
    print("Loading model...")
    model, detector = load_model(args.model, device)
    
    # 寻找视频对
    print("Finding video pairs...")
    video_pairs = find_video_pairs(args.recam_folder, args.original_folder)
    
    print(f"Found {len(video_pairs)} video pairs")
    print("-" * 50)
    
    # 评估每个视频对
    results = []
    
    for i, (recam_video, original_video, video_name) in enumerate(video_pairs):
        print(f"[{i+1}/{len(video_pairs)}] ", end="")
        score = evaluate_video_pair(model, detector, recam_video, original_video, args.num_frames, device, args.model)
        results.append((video_name, score))
        print()
    
    # 输出结果
    print("=" * 50)
    print("FINAL RESULTS:")
    print("=" * 50)
    
    total_score = 0
    for video_name, score in results:
        print(f"{video_name}: {score:.3f}k")
        total_score += score
    
    # 计算平均值
    if len(results) > 0:
        average_score = total_score / len(results)
        print("-" * 50)
        print(f"OVERALL AVERAGE: {average_score:.3f}k")
        print(f"Total videos processed: {len(results)}")
    else:
        print("No valid results to average")

if __name__ == '__main__':
    main()