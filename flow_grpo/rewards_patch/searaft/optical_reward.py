# GT
# === Reward Results ===
# Total Reward: 0.9073

# Detailed Components:
#   temporal_consistency: 0.3820
#   visual_quality: 1.0000
#   flow_ssim: 1.0000
#   endpoint_similarity: 1.0000
#   total_reward: 0.9073
# Time taken: 29.91 seconds

# cam02
# === Reward Results ===
# Total Reward: 0.1863

# Detailed Components:
#   temporal_consistency: 0.4729
#   visual_quality: 0.0034
#   flow_ssim: 0.7609
#   endpoint_similarity: 0.0000
#   total_reward: 0.1863
# Time taken: 29.47 seconds

# cam 03
# === Reward Results ===
# Total Reward: 0.3315

# Detailed Components:
#   temporal_consistency: 0.3895
#   visual_quality: 0.4524
#   flow_ssim: 0.7646
#   endpoint_similarity: 0.0000
#   total_reward: 0.3315
# Time taken: 29.45 seconds






import torch
import torch_npu
import torch.nn.functional as F
import cv2
import re
import json
import numpy as np
import time
import sys

# 添加必要的import（根据您的项目结构调整）
sys.path.append('/home/ma-user/modelarts/user-job-dir/wlh/Code/FlowGRPO/flow_grpo/rewards_patch/searaft/core')
from raft import RAFT
from utils.utils import load_ckpt
import numpy as np
from utils.flow_viz import flow_to_image


# spring-M.json 精简后的关键参数
class SpringArgs:
    def __init__(self):
        self.scale = -1
        self.iters = 4
        self.var_min = 0
        self.var_max = 20
        self.dim = 128
        self.use_var = True
        self.radius = 4
        self.block_dims = [64, 128, 256]
        self.initial_dim = 64
        self.pretrain = 'resnet34'
        self.num_blocks = 2
        
        self.image_size = [540, 960]
        self.scale = -1
        self.batch_size = 32
        self.epsilon = 1e-8
        self.lr = 4e-4
        self.wdecay = 1e-5
        self.dropout = 0
        self.clip = 1.0
        self.gamma = 0.85

SPRING_ARGS = SpringArgs()


class FlowConsistencyReward:
    def __init__(self, args, model, device):
        self.args = args
        self.model = model
        self.device = device
        self.model.eval()
    
    def compute_flow_between_frames(self, frame1, frame2):
        """计算两帧之间的光流"""
        with torch.no_grad():
            flow, info = self.calc_flow(self.args, self.model, frame1, frame2)
            return flow, info
    
    def calc_flow(self, args, model, image1, image2):
        """计算光流（从原始代码复制）"""
        img1 = F.interpolate(image1, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
        img2 = F.interpolate(image2, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
        H, W = img1.shape[2:]
        flow, info = self.forward_flow(args, model, img1, img2)
        flow_down = F.interpolate(flow, scale_factor=0.5 ** args.scale, mode='bilinear', align_corners=False) * (0.5 ** args.scale)
        info_down = F.interpolate(info, scale_factor=0.5 ** args.scale, mode='area')
        return flow_down, info_down
    
    def forward_flow(self, args, model, image1, image2):
        """前向计算光流（从原始代码复制）"""
        output = model(image1, image2, iters=args.scale, test_mode=True)
        flow_final = output['flow'][-1]
        info_final = output['info'][-1]
        return flow_final, info_final
    
    def get_heatmap(self, info, args):
        """获取heatmap（从原始代码复制）"""
        raw_b = info[:, 2:]
        log_b = torch.zeros_like(raw_b)
        weight = info[:, :2].softmax(dim=1)              
        log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=args.var_max)
        log_b[:, 1] = torch.clamp(raw_b[:, 1], min=args.var_min, max=0)
        heatmap = (log_b * weight).sum(dim=1, keepdim=True)
        return heatmap


class CameraControlRewardSystem:
    def __init__(self, args, flow_model, device, video_layout="cthw"):
        self.flow_reward = FlowConsistencyReward(args, flow_model, device)
        self.device = device
        self.video_layout = video_layout.lower()
    
    def _ensure_time_first(self, video_tensor):
        """
        Convert video tensor to [T, C, H, W] if it is provided as [C, T, H, W].
        """
        if video_tensor is None:
            return None
        if not isinstance(video_tensor, torch.Tensor):
            raise TypeError("Expected video tensor input.")
        if video_tensor.dim() != 4:
            raise ValueError(f"Expected video tensor with 4 dims, got {video_tensor.dim()}")
        
        if self.video_layout == "cthw":
            return video_tensor.permute(1, 0, 2, 3).contiguous()
        if self.video_layout == "tchw":
            return video_tensor
        raise ValueError(f"Unsupported video_layout '{self.video_layout}'. Use 'cthw' or 'tchw'.")
        
    def compute_temporal_consistency(self, target_video):
        """
        计算时序一致性reward
        评估生成的target video的帧间平滑度
        """
        target_video = self._ensure_time_first(target_video)
        rewards = []
        total_frames = target_video.shape[0]
        
        for i in range(total_frames - 1):
            frame1 = target_video[i:i+1].to(self.device)
            frame2 = target_video[i+1:i+2].to(self.device)
            
            # 计算光流和置信度
            flow, info = self.flow_reward.compute_flow_between_frames(frame1, frame2)
            heatmap = self.flow_reward.get_heatmap(info, self.flow_reward.args)
            
            # 时序一致性：光流应该平滑变化
            flow_magnitude = torch.sqrt(flow[:, 0]**2 + flow[:, 1]**2)
            flow_smoothness = torch.exp(-torch.std(flow_magnitude))
            
            # 置信度奖励
            confidence_reward = torch.mean(heatmap)
            
            # 组合奖励
            temporal_reward = 0.7 * flow_smoothness + 0.3 * confidence_reward
            rewards.append(temporal_reward.item())
        
        return np.mean(rewards)
    
    def compute_visual_quality_reward(self, target_video, gt_target_video):
        """
        计算视觉质量reward
        可选：与GT视频比较
        """
        target_video = self._ensure_time_first(target_video)
        gt_target_video = self._ensure_time_first(gt_target_video)

        rewards = []
        # 如果有GT视频，计算与GT的光流相似度
        for i in range(min(target_video.shape[0], gt_target_video.shape[0]) - 1):
            # target video的光流
            t_frame1 = target_video[i:i+1].to(self.device)
            t_frame2 = target_video[i+1:i+2].to(self.device)
            t_flow, _ = self.flow_reward.compute_flow_between_frames(t_frame1, t_frame2)
            
            # GT video的光流
            g_frame1 = gt_target_video[i:i+1].to(self.device)
            g_frame2 = gt_target_video[i+1:i+2].to(self.device)
            g_flow, _ = self.flow_reward.compute_flow_between_frames(g_frame1, g_frame2)
            
            # 计算相似度
            similarity = F.cosine_similarity(t_flow.flatten(), g_flow.flatten(), dim=0)
            rewards.append(similarity.item())

        return (np.mean(rewards)+1)/2
    
    def compute_endpoint_error(self, output_target_video, gt_target_video, normalize=True, scale=5.0):
        """
        计算生成视频与GT视频之间的平均Endpoint Error (EPE)
        
        Args:
            output_target_video: 生成的视频
            gt_target_video: GT视频
            normalize: 是否归一化EPE到[0,1]范围
            scale: 归一化时的缩放因子，用于控制归一化曲线的陡峭程度
        """
        output_target_video = self._ensure_time_first(output_target_video)
        gt_target_video = self._ensure_time_first(gt_target_video)

        total_frames = min(output_target_video.shape[0], gt_target_video.shape[0])
        if total_frames < 2:
            return float('nan')
        
        epe_values = []
        gt_flow_magnitudes = []
        
        for i in range(total_frames - 1):
            gen_f1 = output_target_video[i:i+1].to(self.device)
            gen_f2 = output_target_video[i+1:i+2].to(self.device)
            gt_f1 = gt_target_video[i:i+1].to(self.device)
            gt_f2 = gt_target_video[i+1:i+2].to(self.device)
            
            gen_flow, _ = self.flow_reward.compute_flow_between_frames(gen_f1, gen_f2)
            gt_flow, _ = self.flow_reward.compute_flow_between_frames(gt_f1, gt_f2)
            
            diff = gen_flow - gt_flow
            epe = torch.norm(diff, dim=1).mean()
            epe_values.append(epe.item())
            
            # 计算GT光流的幅度，用于相对归一化
            if normalize:
                gt_flow_mag = torch.norm(gt_flow, dim=1).mean()
                gt_flow_magnitudes.append(gt_flow_mag.item())
        
        raw_epe = float(np.mean(epe_values)) if epe_values else float('nan')
        
        if not normalize or np.isnan(raw_epe):
            return raw_epe
        
        # 归一化方法1: 基于GT光流幅度的相对归一化
        # 将EPE相对于GT光流幅度进行归一化，然后映射到[0,1]范围
        if gt_flow_magnitudes:
            avg_gt_flow_mag = np.mean(gt_flow_magnitudes)
            if avg_gt_flow_mag > 1e-6:
                relative_epe = raw_epe / (avg_gt_flow_mag + 1e-6)
                # 使用1-exp(-x)函数将相对EPE映射到[0,1)，保持误差语义（值越大误差越大）
                # scale控制归一化曲线的陡峭程度，scale越大曲线越平缓
                normalized_epe = 1.0 - np.exp(-relative_epe * scale)
                return normalized_epe
        
        # 归一化方法2: 直接使用固定scale的指数衰减归一化（fallback）
        # 使用1-exp(-x/scale)将原始EPE映射到[0,1)，保持误差语义
        # 当epe=0时，normalized_epe=0（完美匹配）
        # 当epe很大时，normalized_epe接近1（误差很大）
        normalized_epe = 1.0 - np.exp(-raw_epe / scale)
        return normalized_epe

    def compute_flow_ssim(self, output_target_video, gt_target_video):
        """
        计算生成视频与GT视频之间的光流与heatmap的SSIM指标
        返回 (flow_ssim, heatmap_ssim)
        """
        output_target_video = self._ensure_time_first(output_target_video)
        gt_target_video = self._ensure_time_first(gt_target_video)

        total_frames = min(output_target_video.shape[0], gt_target_video.shape[0])
        if total_frames < 2:
            return float('nan'), float('nan')

        flow_ssim_values = []
        heatmap_ssim_values = []

        def normalize_flow(flow_tensor):
            # flow_tensor: [B,2,H,W]
            max_val = torch.max(torch.abs(flow_tensor))
            if max_val < 1e-8:
                return torch.zeros_like(flow_tensor)
            flow_norm = flow_tensor / (max_val + 1e-8)
            return (flow_norm + 1.0) * 0.5  # map to [0,1]

        def normalize_heatmap(heat_tensor):
            # heat_tensor: [B,1,H,W]
            min_val = torch.min(heat_tensor)
            max_val = torch.max(heat_tensor)
            if (max_val - min_val) < 1e-8:
                return torch.zeros_like(heat_tensor)
            return (heat_tensor - min_val) / (max_val - min_val + 1e-8)

        for i in range(total_frames - 1):
            gen_f1 = output_target_video[i:i+1].to(self.device)
            gen_f2 = output_target_video[i+1:i+2].to(self.device)
            gt_f1 = gt_target_video[i:i+1].to(self.device)
            gt_f2 = gt_target_video[i+1:i+2].to(self.device)

            gen_flow, gen_info = self.flow_reward.compute_flow_between_frames(gen_f1, gen_f2)
            gt_flow, gt_info = self.flow_reward.compute_flow_between_frames(gt_f1, gt_f2)

            gen_flow_norm = normalize_flow(gen_flow)
            gt_flow_norm = normalize_flow(gt_flow)
            flow_ssim = self._ssim_map(gen_flow_norm, gt_flow_norm)
            flow_ssim_values.append(flow_ssim.mean().item())

            gen_heat = self.flow_reward.get_heatmap(gen_info, self.flow_reward.args)
            gt_heat = self.flow_reward.get_heatmap(gt_info, self.flow_reward.args)
            gen_heat_norm = normalize_heatmap(gen_heat)
            gt_heat_norm = normalize_heatmap(gt_heat)
            heat_ssim = self._ssim_map(gen_heat_norm, gt_heat_norm)
            heatmap_ssim_values.append(heat_ssim.mean().item())

        flow_ssim_mean = float(np.mean(flow_ssim_values))
        heat_ssim_mean = float(np.mean(heatmap_ssim_values))
        return flow_ssim_mean*0.5 + heat_ssim_mean*0.5

    def _ssim_map(self, img1, img2, window_size=11, val_range=1.0):
        # img1/img2: [B,C,H,W] or [C,H,W], values in [0,1] (or any comparable range)
        # returns map per pixel averaged to scalar per batch
        import torch.nn.functional as F
        # make sure we work with 4D tensors (batch dimension)
        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
        if img2.dim() == 3:
            img2 = img2.unsqueeze(0)
        if img1.dim() != 4 or img2.dim() != 4:
            raise ValueError("ssim_map expects tensors with shape [B,C,H,W] or [C,H,W].")
        C1 = (0.01 * val_range) ** 2
        C2 = (0.03 * val_range) ** 2
        # gaussian window
        gauss = torch.Tensor([np.exp(-(x- (window_size//2))**2/float(2*1.5**2)) for x in range(window_size)])
        gauss = gauss/gauss.sum()
        _1D = gauss.unsqueeze(1)
        _2D = gauss[:,None] @ gauss[None,:]
        window = _2D.to(img1.device).unsqueeze(0).unsqueeze(0)
        C = img1.shape[1]
        window = window.repeat(C, 1, 1, 1)
        # compute per-channel means
        mu1 = F.conv2d(img1, window, groups=C, padding=window_size//2)
        mu2 = F.conv2d(img2, window, groups=C, padding=window_size//2)
        mu1_sq = mu1*mu1
        mu2_sq = mu2*mu2
        mu1_mu2 = mu1*mu2
        sigma1_sq = F.conv2d(img1*img1, window, groups=C, padding=window_size//2) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, groups=C, padding=window_size//2) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, groups=C, padding=window_size//2) - mu1_mu2
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        # mean over channels
        return ssim_map.mean(dim=(1,2,3))


    def compute_total_reward(self, source_video, target_camera_extrinsics, output_target_video, gt_target_video=None, weights=None):
        """
        计算总reward
        
        Args:
            source_video: 源视频 [T, C, H, W]
            target_camera_extrinsics: 目标相机外参序列 [T, 4, 4]
            output_target_video: 生成的target video [T, C, H, W]  
            gt_target_video: 可选的GT target video [T, C, H, W]
            weights: 各reward分量的权重
            
        Returns:
            total_reward: 总reward
            reward_components: 各分量reward
        """
        
        if weights is None:
            # weights = {'temporal': 0.15, 'quality': 0.35, 'flow_ssim': 0.15, 'epe': 0.35}
            weights = {'temporal': 0.3, 'quality': 0.35, 'flow_ssim': 0.35}
        
        # 计算各分量reward
        temporal_reward = self.compute_temporal_consistency(output_target_video)
        quality_reward = self.compute_visual_quality_reward(output_target_video, gt_target_video)
        # epe_similarity = 1.0 -self.compute_endpoint_error(output_target_video, gt_target_video)
        flow_ssim_metric = self.compute_flow_ssim(output_target_video, gt_target_video)
        
        # 加权组合（所有指标都是越大越好，直接相加）
        # total_reward = weights['temporal'] * temporal_reward + weights['quality'] * quality_reward + weights['flow_ssim'] * flow_ssim_metric + weights['epe'] * epe_similarity
        total_reward = weights['temporal'] * temporal_reward + weights['quality'] * quality_reward + weights['flow_ssim'] * flow_ssim_metric
        reward_components = {
            'temporal_consistency': temporal_reward,
            'visual_quality': quality_reward,
            'flow_ssim': flow_ssim_metric,
            # 'endpoint_similarity': epe_similarity,  # 相似度值（越大越好）
            'total_reward': total_reward
        }
        
        return total_reward, reward_components


def load_video_frames(video_path, target_size=(540, 960)):
    """加载视频帧并预处理"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 调整大小（如果需要）
        if frame.shape[:2] != target_size:
            frame = cv2.resize(frame, (target_size[1], target_size[0]))
        
        # BGR转RGB并归一化
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        
        frames.append(frame)
    
    cap.release()
    
    # 转换为torch tensor [T, C, H, W]
    frames_tensor = torch.tensor(np.stack(frames)).permute(0, 3, 1, 2)
    return frames_tensor

def parse_extrinsic_string(extrinsic_str):
    """解析外参矩阵字符串为数值数组"""
    # 移除多余的引号和空格
    extrinsic_str = extrinsic_str.strip().strip('"')
    
    # 使用正则表达式提取所有数字
    numbers = re.findall(r'[-+]?\d*\.?\d+', extrinsic_str)
    
    # 应该有16个数字（4x4矩阵）
    if len(numbers) != 16:
        raise ValueError(f"Expected 16 numbers for 4x4 matrix, got {len(numbers)}")
    
    # 转换为float并reshape为4x4矩阵
    matrix = np.array([float(x) for x in numbers]).reshape(4, 4)
    return matrix

def load_camera_extrinsics(json_path, num_frames=None, camera_id="cam01"):
    """加载相机外参数据 - 修复版本"""
    with open(json_path, 'r') as f:
        extrinsics_data = json.load(f)
    
    extrinsics_list = []
    
    # 按帧号排序
    frame_keys = sorted(extrinsics_data.keys(), key=lambda x: int(x.replace('frame', '')))
    
    for frame_key in frame_keys:
        if num_frames and len(extrinsics_list) >= num_frames:
            break
            
        frame_data = extrinsics_data[frame_key]
        
        # 获取指定摄像头的矩阵字符串
        if camera_id not in frame_data:
            raise ValueError(f"Camera {camera_id} not found in frame {frame_key}")
        
        extrinsic_str = frame_data[camera_id]
        
        # 解析字符串为矩阵
        extrinsic_matrix = parse_extrinsic_string(extrinsic_str)
        
        # 转换为torch tensor
        extrinsic_tensor = torch.tensor(extrinsic_matrix, dtype=torch.float32)
        extrinsics_list.append(extrinsic_tensor)
    
    # 堆叠成 [T, 4, 4] 形状
    return torch.stack(extrinsics_list)


def optical_eval(output_target_video, gt_target_video, min_frames=81, device='npu', video_layout = "cthw", model=None, reward_system=None):
    # 初始化参数和模型
    args = SPRING_ARGS
    device = torch.device(device)
    
    # 如果模型未提供，则创建新模型（用于向后兼容）
    if model is None:
        # 加载光流模型
        model = RAFT(args)
        load_ckpt(model, '/home/ma-user/modelarts/user-job-dir/wlh/Model/SeaRaft/Tartan-C-T-TSKH-spring540x960-M.pth')
        # 同步设备操作，避免NPU设备冲突
        if hasattr(torch, 'npu') and torch.npu.is_available():
            torch.npu.synchronize()
        model = model.to(device)
        if hasattr(torch, 'npu') and torch.npu.is_available():
            torch.npu.synchronize()
        model.eval()
    
    # 如果reward系统未提供，则创建新的（用于向后兼容）
    if reward_system is None:
        reward_system = CameraControlRewardSystem(args, model, device, video_layout=video_layout)
    
    # 加载源视频
    source_video = None     # 暂时不用原视频
    # source_video = load_video_frames("source_video.mp4")
    # print(f"Source video loaded: {source_video.shape}")
    
    
    # 加载相机外参
    camera_extrinsics = None    #暂时不用相机外参
    # camera_extrinsics = load_camera_extrinsics("camera_extrinsics.json", num_frames=min(len(source_video), len(output_target_video)))
    # print(f"Camera extrinsics loaded: {camera_extrinsics.shape}")
    
    

    frame_axis = 1 if video_layout == "cthw" else 0
    max_available_frames = min(
        min_frames,
        output_target_video.shape[frame_axis],
        gt_target_video.shape[frame_axis]
    )

    if frame_axis == 1:
        output_target_video = output_target_video[:, :max_available_frames]
        gt_target_video = gt_target_video[:, :max_available_frames]
    else:
        output_target_video = output_target_video[:max_available_frames]
        gt_target_video = gt_target_video[:max_available_frames]
    
    # print(f"Using {min_frames} frames for evaluation")
    
    timestamp = time.time()
    # 计算reward
    total_reward, reward_components = reward_system.compute_total_reward(
        source_video=source_video,
        target_camera_extrinsics=camera_extrinsics,
        output_target_video=output_target_video,
        gt_target_video=gt_target_video
    )
    
    # 输出结果
    print("\n=== Reward Results ===")
    print(f"Total Reward: {total_reward:.4f}")
    print("\nDetailed Components:")
    for component, value in reward_components.items():
        print(f"  {component}: {value:.4f}")
    
    timestamp = time.time() - timestamp
    print(f"Time taken: {timestamp:.2f} seconds")

    return float(total_reward)


def optical_eval_with_details(output_target_video, gt_target_video, min_frames=81, device='npu', video_layout = "cthw", model=None, reward_system=None):
    """
    与optical_eval相同，但返回详细指标
    返回: (total_reward, reward_components)
    """
    # 初始化参数和模型
    args = SPRING_ARGS
    device = torch.device(device)
    
    # 如果模型未提供，则创建新模型（用于向后兼容）
    if model is None:
        # 加载光流模型
        model = RAFT(args)
        load_ckpt(model, '/home/ma-user/modelarts/user-job-dir/wlh/Model/SeaRaft/Tartan-C-T-TSKH-spring540x960-M.pth')
        # 同步设备操作，避免NPU设备冲突
        if hasattr(torch, 'npu') and torch.npu.is_available():
            torch.npu.synchronize()
        model = model.to(device)
        if hasattr(torch, 'npu') and torch.npu.is_available():
            torch.npu.synchronize()
        model.eval()
    
    # 如果reward系统未提供，则创建新的（用于向后兼容）
    if reward_system is None:
        reward_system = CameraControlRewardSystem(args, model, device, video_layout=video_layout)
    
    # 加载源视频
    source_video = None     # 暂时不用原视频
    camera_extrinsics = None    #暂时不用相机外参

    frame_axis = 1 if video_layout == "cthw" else 0
    max_available_frames = min(
        min_frames,
        output_target_video.shape[frame_axis],
        gt_target_video.shape[frame_axis]
    )

    if frame_axis == 1:
        output_target_video = output_target_video[:, :max_available_frames]
        gt_target_video = gt_target_video[:, :max_available_frames]
    else:
        output_target_video = output_target_video[:max_available_frames]
        gt_target_video = gt_target_video[:max_available_frames]
    
    timestamp = time.time()
    # 计算reward
    total_reward, reward_components = reward_system.compute_total_reward(
        source_video=source_video,
        target_camera_extrinsics=camera_extrinsics,
        output_target_video=output_target_video,
        gt_target_video=gt_target_video
    )
    
    return float(total_reward), reward_components
    

# 如果直接运行这个文件
if __name__ == "__main__":
    # 加载生成的target video
    output_target_video = load_video_frames("cam03.mp4")
    print(f"Output target video loaded: {output_target_video.shape}")

    # 加载GT target video
    gt_target_video = load_video_frames("GT_cam01.mp4")
    print(f"GT target video loaded: {gt_target_video.shape}")

    # 注意这里的load_video_frames返回的是[T, C, H, W]格式，与recam代码里的[C, T, H, W]格式不同
    # 所以使用 video_layout="tchw" 来匹配 [T, C, H, W] 格式
    print("Computing optical reward...")
    reward = optical_eval(
        output_target_video, 
        gt_target_video, 
        min_frames=min(output_target_video.shape[0], gt_target_video.shape[0]), 
        device='npu', 
        video_layout="tchw"
    )
        