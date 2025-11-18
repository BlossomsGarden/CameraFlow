"""
DISCLAIMER

FVD is implemented on https://github.com/ragor114/PyTorch-Frechet-Video-Distance
FVD-V is reproduced based on paper SVD4D (https://arxiv.org/pdf/2407.17470) I guess no official code is available?
"""

import numpy as np
import scipy
from torch.utils.data import DataLoader, TensorDataset
import torch, torch_npu
import hashlib
import os
import glob
import requests
import re
import html
import io
import uuid


"""
This function is used to first extract feature representation vectors of the videos using a pretrained model
Then the mean and covariance of the representation vectors are calculated and returned
"""
def compute_feature_stats(data, detector_url, detector_kwargs, batch_size, max_items, device):
    # if wanted reduce the number of elements used for calculating the FVD
    num_items = len(data)
    if max_items:
        num_items = min(num_items, max_items)
    data = data[:num_items]
    
    detector = torch.jit.load(detector_url).eval().to(device)

    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size)
    all_features = []
    for batch in loader:
        batch = batch[0]
        # if more than 3 channels are available we split the channel dimension into chunks of 3 and concatenate to batch dimension
        if batch.size(1) != 3:
            pad_size = 3 - (batch.size(1) % 3)
            pad = torch.zeros(batch.size(0), pad_size, batch.size(2), batch.size(3), batch.size(4), device=batch.device)
            batch = torch.cat([batch, pad], dim=1)
            batch = torch.cat(torch.chunk(batch, chunks=batch.size(1)//3, dim=1), dim=0)
        batch = batch.to(device)
        # extract feature vector using pretrained model
        features = detector(batch, **detector_kwargs)
        features = features.detach().cpu().numpy()
        all_features.append(features)
    # concatenate batches to one numpy array
    stacked_features = np.concatenate(all_features, axis=0)

    # calculate mean and covariance matrix across the extracted features
    mu = np.mean(stacked_features, axis=0)
    sigma = np.cov(stacked_features, rowvar=False)

    return mu, sigma



"""
This function calculates the Frechet Video Distance of two tensors representing a collection of videos
The input tensors should have shape num_videos x channels x num_frames x width x height
As the calculation of frechet video distance can be expensive max_items can be defined to estimate FVD on a subset
"""
def compute_fvd(y_true: torch.Tensor, y_pred: torch.Tensor, max_items: int, device: torch.device, batch_size: int):
    # 定义文件路径和下载链接
    detector_url = '/data/wlh/.cache/torch/hub/checkpoints/i3d_torchscript.pt'
    download_url = 'https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1'

    # 检测文件是否存在
    if not os.path.exists(detector_url):
        raise FileNotFoundError(f"缺少I3D权重文件：{detector_url}，请于此链接下载：{download_url}")

    detector_kwargs = dict(rescale=True, resize=True, return_features=True) # Return raw features before the softmax layer.

    # calculate the mean and covariance matrix of the representation vectors for ground truth and predicted videos
    mu_true, sigma_true = compute_feature_stats(y_true, detector_url, detector_kwargs, batch_size, max_items, device)
    mu_pred, sigma_pred = compute_feature_stats(y_pred, detector_url, detector_kwargs, batch_size, max_items, device)

    # FVD is calculated as the mahalanobis distance between the representation vector statistics
    m = np.square(mu_pred - mu_true).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_pred, sigma_true), disp=False)
    fvd = np.real(m + np.trace(sigma_pred + sigma_true - s * 2))
    return float(fvd)



"""
FVD-V: Frechet Video Distance for View Consistency
计算每个时间帧上不同视图之间的FVD，用于评估多视图一致性
"""
def compute_feature_stats_for_views_per_frame(data, detector_url, detector_kwargs, batch_size, max_items, device, min_required_T=10):
    """
    为 FVD-V 计算单帧的多视角特征统计（I3D 输入需有足够的时间长度以免内部 avg_pool 报错）。
    data: torch.Tensor, shape [N, V, C, H, W]
    min_required_T: 最少时间帧长度（推荐 16）
    """
    num_items = len(data)
    if max_items:
        num_items = min(num_items, max_items)
    data = data[:num_items]

    detector = torch.jit.load(detector_url).eval().to(device)

    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size)
    all_features = []

    for batch in loader:
        batch = batch[0]  # [batch_size, V, C, H, W]
        B, V, C, H, W = batch.shape

        # reshape -> treat each view as an independent short-video with temporal dim = 1
        batch = batch.reshape(B * V, C, 1, H, W)  # [B*V, C, 1, H, W]
        cur_T = batch.size(2)

        # 若时间维度不足，按最小需求扩展（repeat），保证 I3D 内部下采样后仍 >= kernel
        if cur_T < min_required_T:
            # 需要重复的倍数
            repeats = int(np.ceil(min_required_T / cur_T))
            batch = batch.repeat(1, 1, repeats, 1, 1)  # now T >= min_required_T
            # 如果超出，则截取前 min_required_T 帧
            if batch.size(2) > min_required_T:
                batch = batch[:, :, :min_required_T, :, :]

        # 便于调试，打印一次形状（仅在第一次批次）
        # 注意：实际运行中可注释掉以减少IO
        print(f"[DEBUG] detector input shape: {batch.shape} (B*V, C, T, H, W)")

        batch = batch.to(device)

        # 提取特征
        with torch.no_grad():
            features = detector(batch, **detector_kwargs)
        features = features.detach().cpu().numpy()
        all_features.append(features)

    stacked_features = np.concatenate(all_features, axis=0)
    mu = np.mean(stacked_features, axis=0)
    sigma = np.cov(stacked_features, rowvar=False)
    return mu, sigma


"""
计算 FVD-V (View Consistency Frechet Video Distance)
实现方式严格符合 SVD4D 论文定义：对每个时间帧 t，计算不同视角间的 FVD，并取平均。

参数:
    y_true: 真实视频张量, 形状 [N, T, V, C, H, W]
    y_pred: 生成视频张量, 形状 [N, T, V, C, H, W]
    max_items: 最大处理视频数
    device: torch 设备
    batch_size: 批量大小

返回:
    fvd_v_value: 平均跨视角 FVD 值
"""
def compute_fvd_v(y_true: torch.Tensor, y_pred: torch.Tensor, max_items: int, device: torch.device, batch_size: int):

    detector_url = '/data/wlh/.cache/torch/hub/checkpoints/i3d_torchscript.pt'
    download_url = 'https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1'

    if not os.path.exists(detector_url):
        raise FileNotFoundError(f"缺少 I3D 权重文件：{detector_url}\n请从以下链接下载：{download_url}")

    detector_kwargs = dict(rescale=True, resize=True, return_features=True)

    N, T, V, C, H, W = y_true.shape
    assert V >= 1, f"输入视频的视角数V必须≥1，当前为 {V}"
    assert y_pred.shape == y_true.shape, "y_true 与 y_pred 形状不一致"

    fvd_v_per_frame = []

    for t in range(T):
        # 取当前帧的所有视图
        views_true = y_true[:, t]  # [N, V, C, H, W]
        views_pred = y_pred[:, t]

        if V < 2:
            # 如果某帧只有一个视角，跳过该帧
            print(f"警告: 第 {t} 帧视角数量不足 (V={V})，跳过该帧的FVD计算。")
            continue

        # 提取真实与生成的特征统计
        mu_true, sigma_true = compute_feature_stats_for_views_per_frame(
            views_true, detector_url, detector_kwargs, batch_size, max_items, device
        )
        mu_pred, sigma_pred = compute_feature_stats_for_views_per_frame(
            views_pred, detector_url, detector_kwargs, batch_size, max_items, device
        )

        # 计算该帧的FVD
        m = np.square(mu_pred - mu_true).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_pred, sigma_true), disp=False)
        fvd_t = np.real(m + np.trace(sigma_pred + sigma_true - 2 * s))
        fvd_v_per_frame.append(fvd_t)

    if len(fvd_v_per_frame) == 0:
        print("警告: 所有帧均跳过，无法计算FVD-V。返回1000.0。")
        return 1000.0

    fvd_v_value = float(np.mean(fvd_v_per_frame))
    return fvd_v_value