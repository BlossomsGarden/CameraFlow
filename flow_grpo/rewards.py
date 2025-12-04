from PIL import Image
import io
import math
import numpy as np
import torch
import torch_npu
from collections import defaultdict
from tqdm.auto import tqdm


def _release_modules(*modules):
    """
    Move heavy modules back to CPU and clear accelerator caches so we only keep
    VRAM usage while a reward is actively being computed.
    """
    for module in modules:
        if module is None:
            continue
        try:
            move_fn = getattr(module, "to", None)
            if callable(move_fn):
                move_fn("cpu")
        except Exception:
            pass
        finally:
            del module

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.empty_cache()


def imagereward_score(device):
    from flow_grpo.imagereward_scorer import ImageRewardScorer

    def _fn(images, prompts, gt_video=None, cam_extrinsics=None):
        # 打印确认传入的参数格式
        print(f"[imagereward_score] images 类型: {type(images)}, gt_video: {gt_video}, cam_extrinsics 类型: {type(cam_extrinsics)}")
        scorer = ImageRewardScorer(dtype=torch.float32, device=device)
        try:
            if isinstance(images, torch.Tensor):
                images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
                images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
                images = [Image.fromarray(image) for image in images]
            prompts = [prompt for prompt in prompts]
            scores = scorer(prompts, images)
            return scores, {}
        finally:
            _release_modules(scorer)

    return _fn


def unifiedreward_score_sglang(device, max_concurrent_requests=2):
    """
    使用 SGLang 服务器进行 UnifiedReward 评分
    
    Args:
        device: 设备（此函数中未直接使用，但保持接口一致性）
        max_concurrent_requests: 最大并发请求数，防止服务器端显存爆炸
                                建议值：根据服务器GPU显存大小调整
                                - 24GB显存: 2-4
                                - 48GB显存: 4-8
    """
    import asyncio
    from openai import AsyncOpenAI
    import base64
    from io import BytesIO
    import re 

    def pil_image_to_base64(image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
        base64_qwen = f"data:image;base64,{encoded_image_text}"
        return base64_qwen

    def _extract_scores(text_outputs):
        scores = []
        pattern = r"Final Score:\s*([1-5](?:\.\d+)?)"
        for text in text_outputs:
            match = re.search(pattern, text)
            if match:
                try:
                    scores.append(float(match.group(1)))
                except ValueError:
                    scores.append(0.0)
            else:
                scores.append(0.0)
        return scores

    client = AsyncOpenAI(base_url="http://127.0.0.1:17140/v1", api_key="flowgrpo")
    
    # 创建信号量来限制并发请求数
    semaphore = asyncio.Semaphore(max_concurrent_requests)
    
    async def evaluate_image(prompt, image):
        # 使用信号量控制并发数
        async with semaphore:
            question = f"<image>\nYou are given a text caption and a generated image based on that caption. Your task is to evaluate this image based on two key criteria:\n1. Alignment with the Caption: Assess how well this image aligns with the provided caption. Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n2. Overall Image Quality: Examine the visual quality of this image, including clarity, detail preservation, color accuracy, and overall aesthetic appeal.\nBased on the above criteria, assign a score from 1 to 5 after \'Final Score:\'.\nYour task is provided as follows:\nText Caption: [{prompt}]"
            images_base64 = pil_image_to_base64(image)
            response = await client.chat.completions.create(
                model="UnifiedReward-7b-v1.5",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": images_base64},
                            },
                            {
                                "type": "text",
                                "text": question,
                            },
                        ],
                    },
                ],
                temperature=0,
            )
            return response.choices[0].message.content

    async def evaluate_batch_image(images, prompts):
        tasks = [evaluate_image(prompt, img) for prompt, img in zip(prompts, images)]
        # 使用 asyncio.gather 并发执行，但受 semaphore 控制实际并发数
        results = await asyncio.gather(*tasks)
        return results

    def _fn(images, prompts, gt_video=None, cam_extrinsics=None):
        # 打印确认传入的参数格式
        print(f"[unifiedreward_score_sglang] images 类型: {type(images)}, gt_video: {gt_video}, cam_extrinsics 类型: {type(cam_extrinsics)}")
        # 处理Tensor类型转换
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        
        # 转换为PIL Image并调整尺寸
        images = [Image.fromarray(image).resize((512, 512)) for image in images]

        # 执行异步批量评估（受 max_concurrent_requests 限制）
        text_outputs = asyncio.run(evaluate_batch_image(images, prompts))
        score = _extract_scores(text_outputs)
        score = [sc/5.0 for sc in score]
        return score, {}
    
    return _fn


    
def video_ocr_score(device):
    from flow_grpo.ocr import OcrScorer_video_or_image

    def _fn(images, prompts, gt_video=None, cam_extrinsics=None):
        # 打印确认传入的参数格式
        print(f"[video_ocr_score] images 类型: {type(images)}, gt_video: {gt_video}, cam_extrinsics 类型: {type(cam_extrinsics)}")
        scorer = OcrScorer_video_or_image()
        try:
            if isinstance(images, torch.Tensor):
                if images.dim() == 4 and images.shape[1] == 3:
                    images = images.permute(0, 2, 3, 1) 
                elif images.dim() == 5 and images.shape[2] == 3:
                    images = images.permute(0, 1, 3, 4, 2)
                images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            scores = scorer(images, prompts)
            # change tensor to list
            return scores, {}
        finally:
            scorer_module = getattr(scorer, "model", None)
            _release_modules(scorer_module)

    return _fn


def my_reward():
    """Custom reward function for ReCamMaster training"""
    def _fn(output_video, gt_video, prompts, cam_extrinsics=None):
        # 打印确认传入的参数格式
        print(f"[my_reward] output_video 类型: {type(output_video)}, gt_video: {gt_video}, cam_extrinsics 类型: {type(cam_extrinsics)}")
        # Return placeholder scores (list of zeros with length = batch_size)
        batch_size = len(output_video)
        import random
        scores = [random.random() for _ in range(batch_size)]
        return {
            'scores': scores,
            'details': {}
        }

    return _fn

"""
Dynamic Control————
    temporal_consistency：target video帧间平滑度，接近1好
    visual quality：target <-> GT，cos similarity，接近1好
    Endpoint Error：target <-> GT，EPE，接近0好
"""
def optical_reward(device):
    from flow_grpo.rewards_patch.searaft.optical_reward import optical_eval, SPRING_ARGS, CameraControlRewardSystem
    from flow_grpo.rewards_patch.searaft.optical_reward import load_video_frames
    import sys
    import torch
    
    # 添加必要的import路径
    sys.path.append('/home/ma-user/modelarts/user-job-dir/wlh/Code/FlowGRPO/flow_grpo/rewards_patch/searaft/core')
    from raft import RAFT
    from utils.utils import load_ckpt

    def _fn(output_video, gt_video, prompts, cam_extrinsics=None):
        # 打印确认传入的参数格式
        print(f"[optical_reward] output_video 类型: {type(output_video)}, gt_video: {gt_video}, cam_extrinsics 类型: {type(cam_extrinsics)}")
        if cam_extrinsics is not None:
            if isinstance(cam_extrinsics, torch.Tensor):
                print(f"[optical_reward] cam_extrinsics.shape: {cam_extrinsics.shape}")
        # VGG backbone expects float32 tensors; convert from bf16 if needed
        output_video = output_video.float()
        gt_video = gt_video.float()

        # 在_fn内部初始化模型，避免重复创建导致NPU设备同步冲突
        args = SPRING_ARGS
        model = RAFT(args)
        load_ckpt(model, '/home/ma-user/modelarts/user-job-dir/wlh/Model/SeaRaft/Tartan-C-T-TSKH-spring540x960-M.pth')
        
        # 同步设备操作，避免NPU设备冲突
        device_obj = torch.device(device)
        if hasattr(torch, 'npu') and torch.npu.is_available():
            torch.npu.synchronize()
        model = model.to(device_obj)
        if hasattr(torch, 'npu') and torch.npu.is_available():
            torch.npu.synchronize()
        model.eval()
        
        # 创建reward系统
        reward_system = CameraControlRewardSystem(args, model, device_obj, video_layout="cthw")

        try:
            batch_size = output_video.shape[0]
            scores = []
            details_list = []
            from flow_grpo.rewards_patch.searaft.optical_reward import optical_eval_with_details
            for b in range(batch_size):
                # input (C, T, H, W)
                # 复用已创建的模型和reward系统
                score, reward_components = optical_eval_with_details(
                    output_video[b], 
                    gt_video[b], 
                    min_frames=81, 
                    device=device, 
                    video_layout="cthw",
                    model=model,
                    reward_system=reward_system
                )
                scores.append(float(score))
                details_list.append(reward_components)
            
            # 聚合详细指标（每个batch一个值）
            aggregated_details = {}
            if details_list:
                for key in details_list[0].keys():
                    if key != 'total_reward':  # 跳过total_reward，因为已经有scores了
                        values = [d.get(key, 0.0) for d in details_list if isinstance(d.get(key), (int, float))]
                        if values:
                            # 为每个batch样本分配相同的聚合值
                            aggregated_details[key] = values if len(values) == batch_size else [np.mean(values)] * batch_size
            
            return {
                'scores': scores,
                'details': aggregated_details
            }
        finally:
            # 释放资源，将模型移回CPU并清理缓存
            _release_modules(model, reward_system)

    return _fn

"""
Camera Control————
    SSIM
    LPIPS
"""
def gt_reward(device):
    import torch.nn.functional as F
    from flow_grpo.lpips_score import VGGPerceptual

    def _fn(output_video, gt_video, prompts, cam_extrinsics=None):
        # 打印确认传入的参数格式
        print(f"[gt_reward] output_video 类型: {type(output_video)}, gt_video: {gt_video}, cam_extrinsics 类型: {type(cam_extrinsics)}")
        if cam_extrinsics is not None:
            if isinstance(cam_extrinsics, torch.Tensor):
                print(f"[gt_reward] cam_extrinsics.shape: {cam_extrinsics.shape}")
        # VGG backbone expects float32 tensors and tensors on the same device as the model
        output_video = output_video.float()
        gt_video = gt_video.float()
        
        # Ensure input format is torch.Tensor in [0,1], shape: (B, C, T, H, W)
        # Optional: normalize to [0,1] if looks unnormalized
        if output_video.max() > 1.1:
            output_video = output_video / 255.0
        if gt_video.max() > 1.1:
            gt_video = gt_video / 255.0

        # --- SSIM helper ---
        def ssim_torch(img1, img2, window_size=11, channel=3):
            # 计算高斯窗口
            def gaussian(window_size, sigma=1.5):
                x = torch.arange(window_size, dtype=torch.float)
                gauss = torch.exp(-((x - window_size // 2) ** 2) / (2 * sigma ** 2))
                return gauss / gauss.sum()

            _1D_window = gaussian(window_size).unsqueeze(1)
            _2D_window = _1D_window @ _1D_window.t()
            window = _2D_window.expand(channel, 1, window_size, window_size).to(img1.device)

            mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
            mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2

            sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
            sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
            sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

            C1 = 0.01**2
            C2 = 0.03**2

            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                    ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

            return ssim_map.mean(dim=(1, 2, 3))  # (N,)

        # print("output_video.device", output_video.device)
        # print("gt_video.device", gt_video.device)
        # print("device", device)
        vggp = VGGPerceptual(device)
        B, C, T, H, W = output_video.shape
        out_flat = output_video.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)
        gt_flat  = gt_video.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)
        ssim_frame = ssim_torch(out_flat, gt_flat, channel=C)    # (B*T,)
        # Use torch mean directly to avoid numpy dispatch issues on tensors
        mean_ssim = ssim_frame.mean().item()
        
        # Ensure LPIPS network is on the correct device as well
        lpips_values = []
        for b in range(B):
            for t in range(T):
                lpips = vggp(output_video[b, :, t], gt_video[b, :, t])
                lpips_values.append(lpips.item())

        mean_lpips = float(np.mean(lpips_values))
        
        # 关键修复：SSIM越高越好，LPIPS越低越好
        # LPIPS值范围是[0, ∞)，需要归一化到[0,1]范围
        # 使用双曲函数归一化：1 / (1 + lpips / scale)
        # 当lpips=0（完美匹配）时，归一化值=1（最高奖励）
        # 当lpips增大时，归一化值平滑衰减，始终在[0,1]范围内，不会出现负值
        # scale控制衰减速度，scale越大衰减越慢
        # 对于LPIPS，通常值在0~2之间，scale=1.0比较合适
        # 如果LPIPS值普遍较大（>2），可以增大scale（如2.0或3.0）
        lpips_scale = 1.0  # 可以根据实际LPIPS值分布调整
        normalized_lpips = 1.0 / (1.0 + mean_lpips / lpips_scale)
        
        # 现在两个指标都是越大越好，且都在[0,1]范围内
        scores = [mean_ssim * 0.7 + normalized_lpips * 0.3] * B

        # 返回详细指标，用于wandb记录
        return {
            'scores': scores,
            'details': {
                'ssim': [mean_ssim] * B,
                'lpips': [mean_lpips] * B,  # 原始LPIPS值
                'normalized_lpips': [normalized_lpips] * B,  # 归一化后的LPIPS值
            }
        }

    return _fn


"""
Video Quality————
    UnifiedReward (TODO)
    CLIP-V
"""
def clip_v_score(device):
    """
    Returns CLIP-V score: Cosine similarity between ground truth and generated corresponding frames.
    All components are normalized to produce a reward in [0, 1].

    Usage: score = clip_v_score(device)(output_video, gt_video, prompts, cam_extrinsics)
    Returns: reward tensor of shape [B, 1]
    """
    import torch
    import torch.nn.functional as F
    import numpy as np
    from flow_grpo.clip_scorer import ClipScorer

    def _fn(output_video, gt_video, prompts, cam_extrinsics=None):
        # 打印确认传入的参数格式
        print(f"[clip_v_score] output_video 类型: {type(output_video)}, gt_video: {gt_video}, cam_extrinsics 类型: {type(cam_extrinsics)}")
        scorer = ClipScorer(device=device)
        try:
            # output_video, gt_video: [B, C, T, H, W], float32, [0,1] or [0,255]
            output_video = output_video.to(device)
            gt_video = gt_video.to(device)
            B, C, T, H, W = output_video.shape

            clip_v_list = []

            for b in range(B):
                # ------ CLIP-V ------
                src_frames = gt_video[b]
                tgt_frames = output_video[b]
                min_frames = min(src_frames.shape[1], tgt_frames.shape[1])
                if min_frames == 0:
                    clip_v_list.append(0.0)
                else:
                    src_frames_ = src_frames[:, :min_frames, :, :]  # [C, min_frames, H, W]
                    tgt_frames_ = tgt_frames[:, :min_frames, :, :]  # [C, min_frames, H, W]
                    src_frames_ = src_frames_.permute(1, 0, 2, 3)  # [min_frames, C, H, W]
                    tgt_frames_ = tgt_frames_.permute(1, 0, 2, 3)
                    with torch.no_grad():
                        src_processed = scorer._process(src_frames_).to(device)
                        tgt_processed = scorer._process(tgt_frames_).to(device)
                        src_feat = scorer.model.get_image_features(pixel_values=src_processed)
                        tgt_feat = scorer.model.get_image_features(pixel_values=tgt_processed)
                        src_feat = F.normalize(src_feat, p=2, dim=1)
                        tgt_feat = F.normalize(tgt_feat, p=2, dim=1)
                        similarities = F.cosine_similarity(src_feat, tgt_feat, dim=1)  # [min_frames]
                        clip_v_list.append(similarities.mean().item())

            # Normalization: CLIP-V cosine similarity is in [-1, 1], map to [0, 1]
            clip_v_arr = torch.tensor(clip_v_list, dtype=torch.float32)
            norm_clip_v = (clip_v_arr + 1.0) / 2.0

            # 返回[B, 1]，与gt_reward一致
            clip_reward = norm_clip_v.view(-1, 1)  # [B, 1]

            return clip_reward
        finally:
            _release_modules(scorer)

    return _fn


def pickscore_score(device):
    from flow_grpo.pickscore_scorer import PickScoreScorer

    def _fn(output_video, gt_video, prompts, cam_extrinsics=None):
        # 打印确认传入的参数格式
        print(f"[pickscore_score] output_video 类型: {type(output_video)}, gt_video: {gt_video}, cam_extrinsics 类型: {type(cam_extrinsics)}")
        scorer = PickScoreScorer(dtype=torch.float32, device=device)
        try:
            # 参照gt_reward，处理B, C, T, H, W视频输入
            B, C, T, H, W = output_video.shape
            output_video = output_video.to(device)

            # 取3帧分别做PickScore，取均值
            frame_indices = [0, min(35, T-1), min(70, T-1)]
            frame_indices = sorted(set(frame_indices))  # 去重并排序，防止帧数不足时重复
            
            # Handle prompts: if single string, repeat for batch
            if isinstance(prompts, str):
                prompt_list = [prompts] * B
            elif isinstance(prompts, (list, tuple)):
                prompt_list = prompts
            else:
                prompt_list = [prompts] * B

            # 收集所有帧的分数
            frame_scores_list = []
            for idx in frame_indices:
                frames = output_video[:, :, idx, :, :]  # [B, C, H, W]
                scores = scorer(prompt=prompt_list, images=frames)  # Returns tensor [B]
                if isinstance(scores, torch.Tensor):
                    frame_scores_list.append(scores)
                else:
                    frame_scores_list.append(torch.tensor(scores, device=device))

            # 对多帧分数取均值: [num_frames, B] -> [B]
            frame_scores = torch.stack(frame_scores_list, dim=0)  # [num_frames, B]
            scores = frame_scores.mean(dim=0)  # [B]
            
            # 返回列表格式，与gt_reward一致
            return scores.cpu().tolist()
        finally:
            _release_modules(scorer)

    return _fn


def cam_score(device, api_url=None, base_port=34567):
    """
    Camera pose estimation reward using DepthAnything3 API.
    Supports multi-node multi-GPU setup where each GPU has its own API server.
    
    Args:
        device: Device (not directly used, kept for interface consistency)
        api_url: URL of the DA3 API server (if None, auto-detects based on LOCAL_RANK)
        base_port: Base port number for API servers (default: 34567, actual port = base_port + LOCAL_RANK)
                   On each node, GPU 0 uses base_port, GPU 1 uses base_port+1, etc.
    
    Returns:
        Function that evaluates camera pose accuracy and returns scores in [0, 5] range.
    """
    import requests
    import json
    import tempfile
    import os
    import numpy as np
    
    # Auto-detect API URL based on LOCAL_RANK (local GPU index on current node)
    # LOCAL_RANK is the local GPU index (0-7) on each node, not the global RANK (0-31)
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None:
        # Fallback: try to extract GPU index from device string if available
        if isinstance(device, (str, torch.device)):
            device_str = str(device)
            if ':' in device_str:
                try:
                    local_rank = int(device_str.split(':')[-1])
                except ValueError:
                    assert False, f"[cam_score] Error: Failed to extract LOCAL_RANK from device string: {device_str}"
            else:
                assert False, f"[cam_score] Error: Failed to extract LOCAL_RANK from device string: {device}"
        else:
            assert False, f"[cam_score] Error: Failed to extract LOCAL_RANK from device string: {device}"
    else:
        local_rank = int(local_rank)
    
    port = base_port + local_rank
    api_url = f"http://localhost:{port}/evaluate_pose"
    print(f"[cam_score] Auto-detected LOCAL_RANK {local_rank}, using API URL: {api_url}")
    
    # Store base URL (without endpoint) for load/unload model calls
    base_url = api_url.replace('/evaluate_pose', '')
    load_model_url = f"{base_url}/load_model"
    unload_model_url = f"{base_url}/unload_model"
    
    def _tensor_to_json_format(cam_extrinsics_tensor, num_frames=21):
        """
        Convert cam_extrinsics tensor [21, 12] to JSON format matching camera_extrinsics.json.
        
        Args:
            cam_extrinsics_tensor: torch.Tensor of shape [21, 12] or [batch_size, 21, 12]
            num_frames: Number of frames (default: 21)
        
        Returns:
            dict: JSON structure matching camera_extrinsics.json format
        """
        cam_extrinsics_tensor = cam_extrinsics_tensor.detach().cpu().float().numpy()
        
        # Convert [21, 12] to [21, 3, 4] (3x4 matrices)
        cam_extrinsics_reshaped = cam_extrinsics_tensor.reshape(num_frames, 3, 4)
        
        # Build JSON structure: {"frame0": "[...]", "frame1": "[...]", ...}
        json_data = {}
        for frame_idx in range(num_frames):
            frame_key = f"frame{frame_idx}"
            
            # Convert 3x4 matrix to string format matching camera_extrinsics.json
            # Format: "[r11 r12 r13 t1] [r21 r22 r23 t2] [r31 r32 r33 t3] [0 0 0 1]"
            matrix_3x4 = cam_extrinsics_reshaped[frame_idx]  # [3, 4]
            
            # Create 4x4 matrix by adding [0, 0, 0, 1] row
            matrix_4x4 = np.zeros((4, 4), dtype=np.float32)
            matrix_4x4[:3, :] = matrix_3x4
            matrix_4x4[3, 3] = 1.0
            
            # Format as string: "[r11 r12 r13 t1] [r21 r22 r23 t2] [r31 r32 r33 t3] [0 0 0 1]"
            row_strings = []
            for row_idx in range(4):
                row = matrix_4x4[row_idx]
                row_str = " ".join([f"{val:.10g}" for val in row])
                row_strings.append(f"[{row_str}]")
            
            # Join rows with space
            matrix_str = " ".join(row_strings) + " "
            
            # Store directly: {"frame0": "[...]", "frame1": "[...]", ...}
            json_data[frame_key] = matrix_str
        
        return json_data
    
    def _fn(output_video, gt_video, prompts, cam_extrinsics=None):
        """
        Evaluate camera pose estimation accuracy.
        
        Args:
            output_video: List of video file paths or torch.Tensor [B, C, T, H, W]
            gt_video: Not used (kept for interface consistency)
            prompts: List of prompts (not used for pose estimation)
            cam_extrinsics: torch.Tensor of shape [batch_size, 21, 12] or [21, 12]
        
        Returns:
            dict: {
                'scores': List of scores in [0, 5] range,
                'details': {
                    'rot_err': List of rotation errors,
                    'trans_err': List of translation errors,
                    'raw_rot_err': List of raw rotation errors (before normalization),
                    'raw_trans_err': List of raw translation errors (before clipping)
                }
            }
        """
        # cam_extrinsics is required for camera pose evaluation
        assert cam_extrinsics is not None, "cam_extrinsics is required for cam_score evaluation"
        
        # Handle video input: could be list of paths or tensor
        video_paths = output_video
        batch_size = len(video_paths)
        
        # Handle cam_extrinsics: could be [batch_size, 21, 12] or [21, 12]
        # print(f"[cam_score] cam_extrinsics 类型: {type(cam_extrinsics)}, cam_extrinsics.shape: {cam_extrinsics.shape}")
        
        # Ensure batch_size matches
        assert cam_extrinsics.shape[0] == batch_size, f"[cam_score] Warning: cam_extrinsics batch size {cam_extrinsics.shape[0]} != video batch size {batch_size}"
        
        # 延迟加载：在计算前加载模型到显存
        # 直接调用load_model端点，服务器端会检查模型是否已加载，避免重复加载
        try:
            print(f"[cam_score] Loading model on server (port {port})...")
            load_response = requests.post(load_model_url, timeout=300)  # Model loading can take time
            load_response.raise_for_status()
            load_result = load_response.json()
            # print(f"[cam_score] Model load response: {load_result}")
        except requests.exceptions.RequestException as e:
            print(f"[cam_score] Warning: Failed to load model, will auto-load on first request: {e}")
        
        scores = []
        rot_errs = []
        trans_errs = []
        raw_rot_errs = []
        raw_trans_errs = []
        
        for video_idx in range(batch_size):
            video_path = video_paths[video_idx]
            
            # Extract camera extrinsics for this video
            # [batch_size, 21, 12] -> [21, 12]
            cam_ext = cam_extrinsics[video_idx]
            
            # Convert cam_extrinsics to JSON format (expects [21, 12])
            json_data = _tensor_to_json_format(cam_ext, num_frames=21)
            
            # Call API with JSON data directly (no need to save file)
            try:
                payload = {
                    "video_path": video_path,
                    "gt_json_data": json_data
                }
                
                response = requests.post(api_url, json=payload, timeout=300)
                response.raise_for_status()
                
                # 打印原始 response 内容
                print(f"[cam_score] ========== Raw API Response (port {port}) ==========")
                print(f"[cam_score] Status Code: {response.status_code}")
                print(f"[cam_score] Headers: {dict(response.headers)}")
                print(f"[cam_score] Raw Text: {response.text}")
                print(f"[cam_score] ==================================================")
                
                result = response.json()
                
                rot_err_mean = float(result['rot_err_mean'])
                trans_err_mean = float(result['trans_err_mean'])
                
                # Store raw errors before any processing
                raw_rot_errs.append(rot_err_mean)
                raw_trans_errs.append(trans_err_mean)
                
                # Handle TransErr > 30: 
                # According to requirements: if all frames have TransErr > 30, set to 30
                # Since API returns mean, we assume API has already filtered frames with TransErr > 30
                # If the mean is still > 30, it means all frames were > 30, so we clip to 30
                if trans_err_mean > 30:
                    print(f"[cam_score] Warning: TransErr {trans_err_mean:.2f} > 30 for video {video_idx}, clipping to 30 (assuming all frames were filtered)")
                    trans_err_mean = 30.0
                
                trans_errs.append(trans_err_mean)
                rot_errs.append(rot_err_mean)
                
            except requests.exceptions.RequestException as e:
                assert False, f"[cam_score] Error calling API for video {video_idx}: {e}"
                # On error, set to maximum penalty
                rot_err_mean = 5.0  # Maximum rotation error (rad)
                trans_err_mean = 30.0  # Maximum translation error
                raw_rot_errs.append(rot_err_mean)
                raw_trans_errs.append(trans_err_mean)
                rot_errs.append(rot_err_mean)
                trans_errs.append(trans_err_mean)
        
        # Convert errors to scores in [0, 5] range
        # Strategy based on observed data:
        # - RotErr: strictly in [0, 1] range, map directly: rot_score = 5 * (1 - rot_err)
        #   (error=0 → score=5, error=1 → score=0)
        # - TransErr: mainly in [0, 2], occasionally larger values
        #   Use robust piecewise mapping: linear in [0, 2], fast decay for >2
        # - Combine: weighted average with equal weights
        
        # RotErr mapping: direct linear mapping from [0, 1] to [5, 0]
        # rot_err = 0 → score = 5 (perfect)
        # rot_err = 1 → score = 0 (worst)
        # Formula: rot_score = 5 * (1 - rot_err)
        
        # TransErr mapping: robust piecewise mapping
        # - Normal range [0, 2]: linear mapping trans_score = 5 * (1 - trans_err / 2)
        #   trans_err = 0 → score = 5 (perfect)
        #   trans_err = 2 → score = 0 (worst in normal range)
        # - Abnormal range (>2): fast exponential decay to handle outliers
        #   Use a very small scale factor to ensure scores are near-zero but maintain
        #   some distinguishability for different outlier values
        #   Formula: trans_score = max(0, epsilon * exp(-(trans_err - 2) / decay_factor))
        #   where epsilon is very small (e.g., 0.1) to ensure rapid decay
        trans_err_normal_max = 2.0  # Normal range maximum
        trans_err_decay_factor = 0.3  # Decay factor for exponential decay (smaller = faster decay)
        trans_err_outlier_scale = 0.1  # Small scale factor for outlier range to maintain distinguishability
        
        # Store individual scores for details
        rot_scores_list = []
        trans_scores_list = []
        
        # Calculate scores for each error type, then combine
        for rot_err, trans_err in zip(rot_errs, trans_errs):
            # RotErr score: direct linear mapping
            # Clamp rot_err to [0, 1] to handle any edge cases
            rot_err_clamped = max(0.0, min(1.0, rot_err))
            rot_score = 5.0 * (1.0 - rot_err_clamped)
            
            # TransErr score: robust piecewise mapping
            if trans_err <= trans_err_normal_max:
                # Normal range [0, 2]: linear mapping
                # trans_err = 0 → score = 5, trans_err = 2 → score = 0
                trans_score = 5.0 * (1.0 - trans_err / trans_err_normal_max)
            else:
                # Abnormal range (>2): fast exponential decay with small scale
                # Use exponential decay to maintain distinguishability while being robust
                # At trans_err=2, we want score ≈ 0 (close to linear part's 0)
                # For trans_err > 2, score decays rapidly but remains slightly positive
                # This allows some distinguishability while being robust to outliers
                excess = trans_err - trans_err_normal_max
                decay = math.exp(-excess / trans_err_decay_factor)
                # Use small scale factor to ensure scores are very small (near-zero)
                # but still maintain some distinguishability for different outlier values
                # At trans_err=2.5: score ≈ 0.1 * exp(-0.5/0.3) ≈ 0.1 * 0.19 ≈ 0.019
                # At trans_err=3.0: score ≈ 0.1 * exp(-1.0/0.3) ≈ 0.1 * 0.036 ≈ 0.0036
                # At trans_err=4.0: score ≈ 0.1 * exp(-2.0/0.3) ≈ 0.1 * 0.0013 ≈ 0.00013
                trans_score = max(0.0, trans_err_outlier_scale * decay)
            
            # Ensure scores are in [0, 5] range
            rot_score = max(0.0, min(5.0, rot_score))
            trans_score = max(0.0, min(5.0, trans_score))
            
            # Store individual scores for details
            rot_scores_list.append(rot_score)
            trans_scores_list.append(trans_score)
            
            # Combine scores: weighted average (equal weights)
            # This gives a score in [0, 5] range where 5 is best and 0 is worst
            combined_score = rot_score*0.8 + trans_score*0.2
            combined_score = max(0.0, min(5.0, combined_score))  # Clamp to [0, 5]
            scores.append(combined_score)
        
        # 计算完成后释放模型显存
        try:
            print(f"[cam_score] Unloading model from server (port {port}) to free GPU memory...")
            unload_response = requests.post(unload_model_url, timeout=60)
            unload_response.raise_for_status()
            # unload_result = unload_response.json()
            # print(f"[cam_score] Model unload response: {unload_result}")
        except requests.exceptions.RequestException as e:
            print(f"[cam_score] Warning: Failed to unload model: {e}")
        
        return {
            'scores': scores,
            'details': {
                'rot_err': rot_scores_list,  # Store rot_score (computed score) instead of error
                'trans_err': trans_scores_list,  # Store trans_score (computed score) instead of error
                'raw_rot_err': raw_rot_errs,  # Keep raw error values from API
                'raw_trans_err': raw_trans_errs  # Keep raw error values from API
            }
        }
    
    return _fn


def unifiedscore(device, api_url=None, base_port=34575):
    """
    UnifiedReward-Think-qwen3vl-8b video quality evaluation using API server.
    Supports multi-node multi-GPU setup where each GPU has its own API server.
    
    Args:
        device: Device (not directly used, kept for interface consistency)
        api_url: URL of the Qwen3VL API server (if None, auto-detects based on LOCAL_RANK)
        base_port: Base port number for API servers (default: 34575, actual port = base_port + LOCAL_RANK)
                   On each node, GPU 0 uses base_port, GPU 1 uses base_port+1, etc.
    
    Returns:
        Function that evaluates video quality and returns scores in [0, 1] range (normalized from [1.0, 5.0]).
    """
    import requests
    import re
    import os
    
    # Auto-detect API URL based on LOCAL_RANK (local GPU index on current node)
    # LOCAL_RANK is the local GPU index (0-7) on each node, not the global RANK (0-31)
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None:
        # Fallback: try to extract GPU index from device string if available
        if isinstance(device, (str, torch.device)):
            device_str = str(device)
            if ':' in device_str:
                try:
                    local_rank = int(device_str.split(':')[-1])
                except ValueError:
                    assert False, f"[unifiedscore] Error: Failed to extract LOCAL_RANK from device string: {device_str}"
            else:
                assert False, f"[unifiedscore] Error: Failed to extract LOCAL_RANK from device string: {device}"
        else:
            assert False, f"[unifiedscore] Error: Failed to extract LOCAL_RANK from device string: {device}"
    else:
        local_rank = int(local_rank)
    
    port = base_port + local_rank
    api_url = f"http://localhost:{port}/evaluate_video"
    print(f"[unifiedscore] Auto-detected LOCAL_RANK {local_rank}, using API URL: {api_url}")
    
    # Store base URL (without endpoint) for load/unload model calls
    base_url = api_url.replace('/evaluate_video', '')
    load_model_url = f"{base_url}/load_model"
    unload_model_url = f"{base_url}/unload_model"
    
    def extract_answer(text):
        """
        从文本中提取 <answer> 标签中的内容
        仿照 api_request_qwen3vl.py 中的实现
        """
        pattern = r'<answer>(.*?)</answer>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
    
    def _fn(output_video, gt_video, prompts, cam_extrinsics=None):
        """
        Evaluate video quality using UnifiedReward-Think-qwen3vl-8b model.
        
        Args:
            output_video: List of video file paths
            gt_video: Not used (kept for interface consistency)
            prompts: List of prompts describing the video content
            cam_extrinsics: Not used (kept for interface consistency)
        
        Returns:
            dict: {
                'scores': List of scores in [0, 1] range (normalized from [1.0, 5.0]),
                'details': {
                    'raw_scores': List of raw scores in [1.0, 5.0] range,
                    'output_texts': List of raw output texts from the model
                }
            }
        """
        # Handle video input: should be list of paths
        video_paths = output_video
        prompt_list = prompts
        batch_size = len(video_paths)
        
        # Ensure batch sizes match
        assert len(prompt_list) == batch_size, f"[unifiedscore] Warning: prompts batch size {len(prompt_list)} != video batch size {batch_size}"
        
        # 延迟加载：在计算前加载模型到显存
        # 直接调用load_model端点，服务器端会检查模型是否已加载，避免重复加载
        try:
            print(f"[unifiedscore] Loading model on server (port {port})...")
            load_response = requests.post(load_model_url, timeout=300)  # Model loading can take time
            load_response.raise_for_status()
            load_result = load_response.json()
            # print(f"[unifiedscore] Model load response: {load_result}")
        except requests.exceptions.RequestException as e:
            print(f"[unifiedscore] Warning: Failed to load model, will auto-load on first request: {e}")
        
        scores = []
        raw_scores = []
        output_texts = []
        
        for video_idx in range(batch_size):
            video_path = video_paths[video_idx]
            prompt = prompt_list[video_idx]
            
            # Call API - 仿照 api_request_qwen3vl.py 的请求方式
            payload = {
                "video_path": video_path,
                "prompt": prompt,
                "fps": 30.0  # Default FPS
            }
            
            # 使用 while True 循环，确保成功提取分数后才退出
            # 在实际使用中，如果第一次就成功，会立即 break
            max_retries = 3  # 最多重试3次
            retry_count = 0
            success = False
            
            while retry_count < max_retries and not success:
                try:
                    response = requests.post(api_url, json=payload, timeout=600)  # Video evaluation can take time
                    response.raise_for_status()
                    
                    result = response.json()
                    output_text = result.get('output_text', '')
                    
                    # 打印完整结果（用于调试）
                    print(f"\n[unifiedscore] Video {video_idx} - Full Evaluation Result:")
                    print(output_text)
                    
                    # 提取并解析分数
                    raw_score = extract_answer(output_text)
                    if raw_score:
                        try:
                            score = float(raw_score)
                            scores.append(score)
                            success = True
                            break
                        except ValueError:
                            print(f"[unifiedscore] Warning: Could not convert answer content to float: {raw_score}")
        
                    
                    # 未能提取到分数，重试
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"[unifiedscore] Warning: Could not extract score from response, retrying ({retry_count}/{max_retries})...")
                    else:
                        print(f"[unifiedscore] Error: Failed to extract score after {max_retries} attempts")
                        # 使用默认最低分数
                        output_texts.append(output_text)
                        raw_scores.append(1.0)  # Minimum score
                        scores.append(0.0)  # Minimum normalized score
                        success = True  # 标记为完成，避免继续重试
                        
                except requests.exceptions.RequestException as e:
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"[unifiedscore] Error calling API for video {video_idx} (attempt {retry_count}/{max_retries}): {e}")
                        print(f"[unifiedscore] Retrying...")
                    else:
                        print(f"[unifiedscore] Error calling API for video {video_idx} after {max_retries} attempts: {e}")
                        # On error, set to minimum score
                        raw_score = 1.0  # Minimum score
                        normalized_score = 0.0  # Minimum normalized score
                        raw_scores.append(raw_score)
                        scores.append(normalized_score)
                        output_texts.append(f"Error: {str(e)}")
                        success = True  # 标记为完成，避免继续重试
        
        # 计算完成后释放模型显存
        try:
            print(f"[unifiedscore] Unloading model from server (port {port}) to free GPU memory...")
            unload_response = requests.post(unload_model_url, timeout=60)
            unload_response.raise_for_status()
            # unload_result = unload_response.json()
            # print(f"[unifiedscore] Model unload response: {unload_result}")
        except requests.exceptions.RequestException as e:
            print(f"[unifiedscore] Warning: Failed to unload model: {e}")
        
        # 只返回数值类型的 details，移除字符串类型（output_texts）以适配 multi_score 接口
        # output_texts 仅用于内部调试，不传递到训练流程
        return {
            'scores': scores,  # List[float], shape: [batch_size]
            'details': {
                'raw_scores': raw_scores,  # List[float], shape: [batch_size], 原始分数 [1.0, 5.0]
            }
        }
    
    return _fn


def multi_score(device, score_dict):
    score_functions = {
        "video_ocr": video_ocr_score,
        "imagereward": imagereward_score,
        "unifiedreward": unifiedreward_score_sglang,
        "clip_v_score": clip_v_score,
        "pick_score": pickscore_score,
        "my_reward": my_reward,
        "optical_reward": optical_reward,
        "gt_reward": gt_reward,
        "cam_score": cam_score,
        "unifiedscore": unifiedscore,
    }
    score_fns={}
    for score_name, weight in score_dict.items():
        score_fns[score_name] = score_functions[score_name](device) if 'device' in score_functions[score_name].__code__.co_varnames else score_functions[score_name]()

    # During training, only the strict reward is needed, and non-strict rewards don't need to be computed, reducing reward calculation time.
    def _fn(output_video, gt_video, prompts, cam_extrinsics=None):        
        total_scores = []
        score_details = {}
        score_sub_details = {}  # 存储每个reward函数的详细指标
        score_items = list(score_dict.items())
        for score_name, weight in tqdm(
            score_items,
            desc="Evaluating rewards",
            total=len(score_items),
            leave=False,
            dynamic_ncols=True,
        ):  
            # print("当前计算的奖励是：", score_name)
            # print("传入的类型output_video.device", output_video.device)
            # print("传入的类型gt_video.device", gt_video.device)
            result = score_fns[score_name](output_video, gt_video, prompts, cam_extrinsics)
            
            scores = result['scores']
            if 'details' in result and result['details']:
                # 存储详细指标，键名为 score_name_sub_metric_name
                for sub_metric_name, sub_values in result['details'].items():
                    detail_key = f"{score_name}_{sub_metric_name}"
                    score_sub_details[detail_key] = sub_values
            
            score_details[score_name] = scores
            weighted_scores = [weight * score for score in scores]
            
            if not total_scores:
                total_scores = weighted_scores
            else:
                total_scores = [total + weighted for total, weighted in zip(total_scores, weighted_scores)]
        
        score_details['avg'] = total_scores
        # 将详细指标合并到score_details中
        score_details.update(score_sub_details)
        return score_details

    return _fn


def main():
    import torchvision.transforms as transforms

    image_paths = [
        "nasa.jpg",
    ]

    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor
    ])

    images = torch.stack([transform(Image.open(image_path).convert('RGB')) for image_path in image_paths])
    prompts=[
        'A astronaut’s glove floating in zero-g with "NASA 2049" on the wrist',
    ]
    score_dict = {
        "unifiedreward": 1.0
    }
    # Initialize the multi_score function with a device and score_dict
    device = torch.device("npu" if torch.npu.is_available() else "cpu")
    scoring_fn = multi_score(device, score_dict)
    # Get the scores
    scores = scoring_fn(images, prompts)
    # Print the scores
    print("Scores:", scores)


if __name__ == "__main__":
    main()