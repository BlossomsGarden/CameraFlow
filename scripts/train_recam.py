from collections import defaultdict
import contextlib
import os
os.environ["WANDB_MODE"] = "offline"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
# # 添加以下两行禁用FlashAttention
# os.environ["USE_FLASH_ATTENTION"] = "0"
# os.environ["USE_MEMORY_EFFICIENT_ATTENTION"] = "0"
import sys
sys.path.append('/home/ma-user/modelarts/user-job-dir/wlh/Code/FlowGRPO')
# print(sys.path)
import warnings
warnings.filterwarnings("ignore")
import datetime
from concurrent import futures
import time
import json
import hashlib
import glob
from typing import Optional
from absl import app, flags
# 在文件开头的 imports 部分加入：
from accelerate import Accelerator
from ml_collections import config_flags
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers.utils.torch_utils import is_compiled_module
import numpy as np
import pandas as pd
from PIL import Image
import imageio
from torchvision.transforms import v2
from einops import rearrange
import torchvision
from recam.diffsynth import WanVideoReCamMasterPipeline, ModelManager

import flow_grpo.prompts
import flow_grpo.rewards
from flow_grpo.stat_tracking import PerPromptStatTracker
from flow_grpo.diffusers_patch.wan_pipeline_with_logprob import sde_step_with_logprob
from flow_grpo.diffusers_patch.wan_prompt_embedding import encode_prompt

import torch
import torch_npu
import torch.nn as nn
import wandb
from functools import partial
import tqdm
import tempfile
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, PeftModel
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from flow_grpo.ema import EMAModuleWrapper
import re
import copy


tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)

# Camera class for pose handling
class Camera(object):
    def __init__(self, c2w):
        c2w_mat = np.array(c2w).reshape(4, 4)
        self.c2w_mat = c2w_mat
        self.w2c_mat = np.linalg.inv(c2w_mat)


# ReCamMaster Dataset (modified from train_recammaster_without_grpo.py TensorDataset)
class ReCamMasterDataset(Dataset):
    """Dataset for ReCamMaster video generation training
    
    Loads preprocessed latents from .tensors.pth files and constructs training batches
    with target and condition camera pairs.
    """
    
    def __init__(self, dataset_path, metadata_file_name=None, split='train', steps_per_epoch=None):
        # Load metadata
        if metadata_file_name is None:
            metadata_file_name = f"metadata-{split}.csv"
        metadata_path = os.path.join(dataset_path, metadata_file_name) if not os.path.isabs(metadata_file_name) else metadata_file_name
        metadata = pd.read_csv(metadata_path)

        # Build paths to .tensors.pth files and store corresponding text prompts
        self.path = []
        self.path_to_text = {}
        train_root = os.path.join(dataset_path, "train")
        for _, row in metadata.iterrows():
            file_name = row["file_name"]
            text = row.get("text", "")
            if pd.isna(text):
                text = ""
            tensor_path = os.path.join(train_root, file_name) + ".tensors.pth"
            if os.path.exists(tensor_path):
                self.path.append(tensor_path)
                self.path_to_text[tensor_path] = text
        
        # For training, use steps_per_epoch; for validation, use actual dataset length
        self.steps_per_epoch = steps_per_epoch if steps_per_epoch is not None else len(self.path)

    def parse_matrix(self, matrix_str):
        """Parse camera extrinsics matrix from string format"""
        rows = matrix_str.strip().split('] [')
        matrix = []
        for row in rows:
            row = row.replace('[', '').replace(']', '')
            matrix.append(list(map(float, row.split())))
        return np.array(matrix)

    def get_relative_pose(self, cam_params):
        """Compute relative pose from condition camera first frame to target camera frames"""
        abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
        abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
        target_cam_c2w = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        abs2rel = target_cam_c2w @ abs_w2cs[0]
        ret_poses = [target_cam_c2w, ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
        ret_poses = np.array(ret_poses, dtype=np.float32)
        return ret_poses

    def __getitem__(self, index):
        """
        Returns:
            data['latents']: torch.Size([16, 42, 60, 104]) - concatenated target+condition latents
            data['camera']: torch.Size([21, 12]) - relative pose embeddings
            data['prompt_emb']: dict with 'context' key containing text embeddings
        """
        # 调试信息：打印接收到的索引
        import os
        rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
        # if rank == 0:
        #     print(f"[Dataset __getitem__] Received index: {index}, dataset length: {len(self.path)}")
        
        while True:
            try:
                data = {}
                
                # 1. Use sampler-provided index deterministically.
                #    index may be larger than len(self.path) (some samplers do); use modulo.
                data_id = int(index) % len(self.path)
                path_tgt = self.path[data_id]
                data_tgt = torch.load(path_tgt, weights_only=True, map_location="cpu")

                # 2. Extract source video's cam_id
                match = re.search(r'cam(\d+)', path_tgt)
                tgt_idx = int(match.group(1))  # 1~10（注意，不是0~9）
                
                # 3. Choose condition cam_id (deterministicly choose next camera)
                #    DistributedKRepeatSampler 只产生一个data_index的重复采样，不能同时产生video_id和cam_id两个值
                #    那干脆就直接用它来取视频吧，然后直接取下一个相机作为条件，应该就实现了一个输入同一个index就能获取到相同的两个条件
                cond_idx = tgt_idx + 1 if tgt_idx < 10 else 1
                
                # 4. Load condition camera latents
                path_cond = re.sub(r'cam(\d+)', f'cam{cond_idx:02}', path_tgt)
                data_cond = torch.load(path_cond, weights_only=True, map_location="cpu")
                
                # 5. Concatenate latents (target + condition)
                data['latents'] = torch.cat((data_tgt['latents'], data_cond['latents']), dim=1)
                # Output shape: [16, 42, 60, 104] where 42 = 21 (target) + 21 (condition)
                
                # 6. Load text embeddings and prompt text
                data['prompt_emb'] = data_tgt['prompt_emb']
                data['prompt'] = self.path_to_text.get(path_tgt, "")
                
                # 7. Load camera extrinsics and compute relative poses
                # 7.1 Read camera extrinsics file
                base_path = path_tgt.rsplit('/', 2)[0]  # Get scene directory
                tgt_camera_path = os.path.join(base_path, "cameras", "camera_extrinsics.json")
                try:
                    with open(tgt_camera_path, 'r') as file:
                        cam_data = json.load(file)
                except json.JSONDecodeError as e:
                    print(f"JSON decode error in file: {tgt_camera_path}")
                    print(f"Error details: {e}")
                    raise
                
                # 7.2 Select frames (sample every 4 frames: 81 frames -> 21 frames)
                # This aligns with VAE temporal downsampling
                cam_idx = list(range(81))[::4]  # [0, 4, 8, ..., 80] (21 frames)
                
                # 7.3 Load condition and target camera trajectories
                multiview_c2ws = []
                for view_idx in [cond_idx, tgt_idx]:
                    # Parse c2w matrices for each frame
                    traj = [self.parse_matrix(cam_data[f"frame{idx}"][f"cam{view_idx:02d}"]) 
                            for idx in cam_idx]
                    traj = np.stack(traj).transpose(0, 2, 1)  # Adjust dimensions
                    
                    # Coordinate system transformation: adjust column order and signs
                    c2ws = []
                    for c2w in traj:
                        c2w = c2w[:, [1, 2, 0, 3]]  # Column reordering
                        c2w[:3, 1] *= -1.           # Y-axis flip
                        c2w[:3, 3] /= 100           # Position scaling (cm -> m)
                        c2ws.append(c2w)
                    multiview_c2ws.append(c2ws)
                
                # 7.4 Create Camera objects
                cond_cam_params = [Camera(cam_param) for cam_param in multiview_c2ws[0]]
                tgt_cam_params = [Camera(cam_param) for cam_param in multiview_c2ws[1]]
                
                # 7.5 Compute relative poses
                relative_poses = []
                for i in range(len(tgt_cam_params)):
                    # Compute relative pose from condition camera first frame to target camera frame i
                    relative_pose = self.get_relative_pose([cond_cam_params[0], tgt_cam_params[i]])
                    relative_poses.append(torch.as_tensor(relative_pose)[:,:3,:][1])
                    # Extract 3x4 rotation-translation matrix
                
                # 7.6 Flatten pose embeddings
                pose_embedding = torch.stack(relative_poses, dim=0)  # [21, 3, 4]
                pose_embedding = rearrange(pose_embedding, 'b c d -> b (c d)')  # [21, 12]
                data['camera'] = pose_embedding.to(torch.bfloat16)
                
                # 7.7 Save dataset index as condition identifier for GRPO grouping
                # data_id is the actual dataset index (after modulo), which identifies the unique condition
                # This will be used to group k samples from the same condition
                data['dataset_index'] = data_id
                
                data['metadata'] = {}
                
                break
            except Exception as e:
                print(f"ERROR WHEN LOADING: {e}")
                import traceback
                traceback.print_exc()
                index = random.randrange(len(self.path))
        
        return data
    
    def __len__(self):
        return self.steps_per_epoch


    # 这个函数的作用是将 DataLoader 取到的单个样本列表（examples）打包（collate）成批数据(batch)。
    # 它会把每个样本里的"latents", "camera"等字段分别拼接成一个更大的张量，并且把prompt_emb（提示词特征）字段做合并处理。
    # 这样最终train_dataloader或test_dataloader每次yield出来时，获得的是结构化的batch数据，可直接送入模型训练/推理。
    # 
    # 在数据流通路中，这个函数会作为DataLoader的collate_fn参数。也就是
    # train_dataloader = DataLoader(..., collate_fn=ReCamMasterDataset.collate_fn, ...)
    # 每次DataLoader创建一个batch（通常来自__getitem__的单个样本list）就会自动调用这个collate_fn来聚合数据。
    @staticmethod
    def collate_fn(examples):
        """
        批处理collate函数，将一批样本（examples）合并为可以送给模型的批数据字典。
        在DataLoader加载每一个batch时会被自动调用。
        """
        # 调试信息：打印 collate_fn 接收到的 prompts
        # rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
        # if rank == 0:
        #     collate_prompts = [example.get("prompt", "")[:100] + "..." if len(example.get("prompt", "")) > 100 else example.get("prompt", "") for example in examples]
        #     print(f"[Collate Debug] Received {len(examples)} examples")
        #     print(f"[Collate Debug] Prompts in collate_fn: {collate_prompts}")
        
        # 合并latents、camera等字段
        latents = torch.stack([example["latents"] for example in examples])
        cameras = torch.stack([example["camera"] for example in examples])
        
        # 合并prompt_emb字段，处理dict嵌套情况
        prompt_embs = []
        for example in examples:
            prompt_emb = example["prompt_emb"]
            if isinstance(prompt_emb, dict) and "context" in prompt_emb:
                # context字段可能是list或tensor
                if isinstance(prompt_emb["context"], list):
                    prompt_embs.append(prompt_emb["context"][0])  # 取第一个元素
                else:
                    prompt_embs.append(prompt_emb["context"])
            else:
                prompt_embs.append(prompt_emb)
        prompt_embeds = torch.stack(prompt_embs, dim=0) if len(prompt_embs) > 0 else None

        # 组装batch数据
        batch_data = {
            "latents": latents,    # [batch_size, 16, 42, 60, 104]
            "camera": cameras,     # [batch_size, 21, 12]
            "prompt_emb": {"context": prompt_embeds} if prompt_embeds is not None else {}
        }
        # 打包每个样本的prompt文本，供奖励函数评估等用途
        prompts = [example["prompt"] for example in examples]
        
        # 提取dataset_index作为条件标识符，用于GRPO分组
        # 转换为tensor以便后续处理
        dataset_indices = torch.tensor([example["dataset_index"] for example in examples], dtype=torch.long)
        
        # # 调试信息：打印最终返回的 prompts
        # if rank == 0:
        #     final_prompts = [p[:100] + "..." if len(p) > 100 else p for p in prompts]
        #     print(f"[Collate Debug] Returning prompts: {final_prompts}")
        
        return prompts, batch_data, dataset_indices



# 分布式K重复采样器，用于GRPO算法
# 每个提示词生成K张图片，用于计算组内相对奖励
class DistributedKRepeatSampler(Sampler):
    # 初始化采样器参数，确保总样本数能被K整除
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size  # Batch size per replica
        self.k = k                    # Number of repetitions per sample
        self.num_replicas = num_replicas  # Total number of replicas
        self.rank = rank              # Current replica rank
        self.seed = seed              # Random seed for synchronization
        
        # Compute the number of unique samples needed per iteration
        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, f"k can not divide n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k  # Number of unique samples
        # print("="*100)
        # print("total_samples", self.total_samples)
        # print("k", self.k)
        # print("num_replicas", self.num_replicas)
        # print("batch_size", self.batch_size)
        # print("num_of_unique_samples", self.m)
        # print("="*100)
        self.epoch = 0


    #
    # 每次调用 next(train_iter) 时，生成器会执行一次 yield，然后继续循环
    # 由于 set_epoch 在每次循环中都被调用（第1324行），每次 next() 都会生成新的索引序列
    def __iter__(self):
        while True:
            # Generate a deterministic random sequence to ensure all replicas are synchronized
            # 关键修复：每次迭代时读取当前的 epoch 值，确保使用最新的 epoch
            current_epoch = self.epoch
            g = torch.Generator()
            g.manual_seed(self.seed + current_epoch)
            
            # Randomly select m unique samples
            indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()
            
            # Repeat each sample k times to generate n*b total samples
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            
            # Shuffle to ensure uniform distribution
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]
            
            # Split samples to each replica
            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            
            # # 调试信息：打印更详细的采样信息
            # if self.rank == 0:
            #     print(f"[Sampler Debug] Rank {self.rank} batch indices: {per_card_samples[self.rank]}")
            #     # 打印所有唯一样本的索引，以及打乱后的完整序列（前8个）
            #     print(f"[Sampler Debug] All unique indices: {indices}")
            #     print(f"[Sampler Debug] All shuffled samples: {shuffled_samples}")
            #     print(f"[Sampler Debug] Dataset length: {len(self.dataset)}")
            
            # Return current replica's sample indices
            yield per_card_samples[self.rank]
    
    # 更新epoch以同步随机状态
    def set_epoch(self, epoch):
        self.epoch = epoch  # Used to synchronize random state across epochs
        # 调试信息：确认 epoch 已更新
        if self.rank == 0:
            print(f"[Sampler set_epoch] Updated epoch to {epoch}")



def calculate_zero_std_ratio(prompts, gathered_rewards):
    """
    Calculate the proportion of unique prompts whose reward standard deviation is zero.
    
    Args:
        prompts: List of prompts.
        gathered_rewards: Dictionary containing rewards, must include the key 'ori_avg'.
        
    Returns:
        zero_std_ratio: Proportion of prompts with zero standard deviation.
        prompt_std_devs: Mean standard deviation across all unique prompts.
    """
    # Convert prompt list to NumPy array
    prompt_array = np.array(prompts)
    
    # Get unique prompts and their group information
    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array, 
        return_inverse=True,
        return_counts=True
    )
    
    # Group rewards for each prompt
    grouped_rewards = gathered_rewards['ori_avg'][np.argsort(inverse_indices)]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)

    print("="*100)
    print("unique_prompts", unique_prompts)
    print("inverse_indices", inverse_indices)
    print("counts", counts)
    print("grouped_rewards", grouped_rewards)
    print("split_indices", split_indices)
    print("reward_groups", reward_groups)
    print("="*100)
    
    # Calculate standard deviation for each group
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    
    # Calculate the ratio of zero standard deviation
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)
    
    return zero_std_ratio, prompt_std_devs.mean()


# 扩散模型SDE框架中，给定当前状态转移到下一个状态的概率的对数值
# log p(z_{t-1} | z_t, condition) 概率衡量了从z_t到z_{t-1}这个转移步骤的似然性
#
# 关键差异：
# Flow Matching：
# # 直接学习向量场
# dx/dt = v_θ(x_t, t)  # 学习速度场
# loss = ||v_θ(x_t, t) - u(x_t, t)||^2  # 匹配目标向量场
#
# 扩散模型SDE框架：
# # 学习噪声预测
# dz = f(z,t)dt + g(t)dw  # 前向SDE
# dz = [f(z,t) - g(t)²∇log p_t(z)]dt + g(t)dw  # 反向SDE
# loss = ||ε - ε_θ(z_t, t)||^2  # 预测噪声
#
# 他其实是GRPO里当前策略下某个动作的概率! 策略π(a|s) = 从z_t到z_{t-1}的转移概率
# log π(a|s) = log p(z_{t-1} | z_t, c; θ)
def compute_log_prob_recam(
    transformer,
    pipeline,
    sample,
    j,
    prompt_embeds,
    camera_emb,
    source_latents,
    config,
    negative_prompt_embeds=None,
):
    """
    Compute log probability for ReCamMaster model with camera embeddings and source video.
    """
    device = sample["latents"].device
    model_dtype = pipeline.torch_dtype
    
    # Current latent state z_t and corresponding timestep for every sample in batch
    target_latents = sample["latents"][:, j]  # (batch_size, C, T, H, W)
    next_latents = sample["next_latents"][:, j]
    timestep = sample["timesteps"][:, j]
    
    target_latents_model = target_latents.to(device=device, dtype=model_dtype)
    condition_latents = source_latents.to(device=device, dtype=model_dtype)
    cam_emb = camera_emb.to(device=device, dtype=model_dtype)
    timestep_model = timestep.to(device=device, dtype=model_dtype)
    prompt_embeds = prompt_embeds.to(device=device, dtype=model_dtype)
    negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=model_dtype)
    
    # Prepare latent input: concatenate target latents with source latents along temporal dimension
    latents_input = torch.cat([target_latents_model, condition_latents], dim=2)


    
    if config.train.cfg:
        use_gc = getattr(config.train, "gradient_checkpointing", False)
        use_gc_offload = getattr(config.train, "gradient_checkpointing_offload", False)

        torch.npu.synchronize()
        noise_pred_posi = transformer(
            x=latents_input,
            timestep=timestep_model,
            context=prompt_embeds,
            cam_emb=cam_emb,
            use_gradient_checkpointing=use_gc,
            use_gradient_checkpointing_offload=use_gc_offload,
        )
        torch.npu.synchronize()
        with torch.no_grad():
            torch.npu.synchronize()
            noise_pred_nega = transformer(
                x=latents_input,
                timestep=timestep_model,
                context=negative_prompt_embeds,
                cam_emb=cam_emb
            )
            torch.npu.synchronize()
        noise_pred = noise_pred_nega + config.sample.guidance_scale * (noise_pred_posi - noise_pred_nega)
    else:
        use_gc = getattr(config.train, "gradient_checkpointing", False)
        use_gc_offload = getattr(config.train, "gradient_checkpointing_offload", False)
        noise_pred = transformer(
            x=latents_input,
            timestep=timestep_model,
            context=prompt_embeds,
            cam_emb=cam_emb,
            use_gradient_checkpointing=use_gc,
            use_gradient_checkpointing_offload=use_gc_offload,
        )
    
    # Extract only the target (denoised) portion before computing log-prob and scheduler updates
    tgt_latent_length = target_latents_model.shape[2]
    noise_pred_target = noise_pred[:, :, :tgt_latent_length, ...]
    
    # Ensure scheduler has index_for_timestep method for compatibility with sde_step_with_logprob
    if not hasattr(pipeline.scheduler, "index_for_timestep"):
        add_index_for_timestep_to_scheduler(pipeline.scheduler)
    
    prev_sample, log_prob, prev_sample_mean, std_dev_t = sde_step_with_logprob(
        pipeline.scheduler,
        noise_pred_target.float(),
        timestep.to(dtype=torch.float32, device=device),
        target_latents.float(),
        prev_sample=next_latents.float(),
    )
    
    # Free large intermediates ASAP to reduce peak memory
    del noise_pred, noise_pred_target, latents_input
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    return prev_sample, log_prob, prev_sample_mean, std_dev_t


def add_index_for_timestep_to_scheduler(scheduler):
    """
    Add index_for_timestep method to FlowMatchScheduler to make it compatible with sde_step_with_logprob.
    This method is required by sde_step_with_logprob from wan_pipeline_with_logprob.py.
    """
    def index_for_timestep(self, timestep):
        """
        Find the index of the timestep in self.timesteps.
        Similar to FlowMatchEulerDiscreteScheduler.index_for_timestep.
        """
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        else:
            timestep = torch.tensor(timestep)
        
        # Find the closest timestep index
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        if isinstance(timestep_id, torch.Tensor):
            return timestep_id.item()
        return timestep_id
    
    # Bind the method to the scheduler instance
    import types
    scheduler.index_for_timestep = types.MethodType(index_for_timestep, scheduler)
    return scheduler


def save_video(frames, save_path, fps, quality=9, ffmpeg_params=None):
    writer = imageio.get_writer(save_path, fps=fps, quality=quality, ffmpeg_params=ffmpeg_params)
    for frame in tqdm(frames, desc="Saving video"):
        frame = np.array(frame)
        writer.append_data(frame)
    writer.close()


def recam_pipeline_with_logprob(
    pipeline,
    prompt_embeds,
    negative_prompt_embeds,
    all_latents,
    target_camera,
    num_inference_steps=50,
    guidance_scale=5.0,
    output_type="tensor",
    height=480,
    width=832,
    num_frames=81,
    tiled=True,
    tile_size=(30, 52),
    tile_stride=(15, 26),
):
    """
    ReCamMaster pipeline with log probability tracking for GRPO training.
    Returns: (videos, all_latents, all_log_probs)
    """
    device = all_latents.device
    # config.sample.train_batch_size
    # Use batch_size from all_latents to ensure consistency with source_latents
    # The last batch may not be full batch_size, so we need to use the actual batch size from data
    batch_size = all_latents.shape[0]
    # logger.info(f"batch_size from all_latents: {batch_size}, negative_prompt_embeds.shape[0]: {negative_prompt_embeds.shape[0]}")
    
    # Ensure prompt embeddings match the actual batch size
    if prompt_embeds.shape[0] != batch_size:
        prompt_embeds = prompt_embeds[:batch_size]
    if negative_prompt_embeds.shape[0] != batch_size:
        negative_prompt_embeds = negative_prompt_embeds[:batch_size]
    if target_camera.shape[0] != batch_size:
        target_camera = target_camera[:batch_size]
    
    
    # Extract ReCamMaster data: latents (concatenated target+condition) and camera
    source_latents = all_latents[:, :, 21:, ...]  # (batch_size, 16, 21, 60, 104)
    target_latents = all_latents[:, :, :21, ...]   # (batch_size, 16, 21, 60, 104)
    
    # Ensure source_latents and target_latents are on the correct device
    # This is important because in the second epoch, tensors might be on a different device
    # due to previous model offloading operations
    source_latents = source_latents.to(device=device)
    target_latents = target_latents.to(device=device)
    
    # Encode source video to latents
    tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
    
    # Prepare scheduler
    pipeline.scheduler.set_timesteps(num_inference_steps, denoising_strength=1.0, shift=5.0)
    timesteps = pipeline.scheduler.timesteps
    # print("看看你自动生成的timesteps是什么样子的", timesteps)
    # print("看看你自动生成的timesteps有多少步", len(timesteps))

    # Prepare latent variables (noise)
    # ReCamMaster uses generate_noise method similar to inference code
    # Use 16 channels (standard for Wan video model)
    # 准备潜在变量（噪声）
    num_channels_latents = 16
    latents = pipeline.generate_noise(
        (batch_size, num_channels_latents, (num_frames - 1) // 4 + 1, height//8, width//8),
        seed=None,
        device=device,
        dtype=torch.float32
    )
    latents = latents.to(dtype=pipeline.torch_dtype, device=device)
    
    # Process target camera
    # 处理目标相机参数
    cam_emb = target_camera.to(dtype=pipeline.torch_dtype, device=device)
    
    all_latents = [latents]
    all_log_probs = []

    # Denoising loop (去噪循环)
    # 
    # timesteps 含义解释：
    # - timesteps 是一个时间步序列，表示从高噪声到低噪声的去噪过程
    # - 例如：[1000, 980, 960, ..., 20, 0]，数值越大表示噪声越多
    # - 循环从高噪声（t=1000）开始，逐步去噪到低噪声（t=0），最终生成清晰视频
    # - num_inference_steps 控制总步数，更多步数通常质量更好但速度更慢
    #
    # 循环作用：
    # 1. 每一步使用 DIT 模型预测当前噪声（noise prediction）
    # 2. 使用调度器（scheduler）根据预测的噪声更新 latents（去噪一步）
    # 3. 计算这一步的对数概率 log_prob（用于 GRPO 强化学习训练）
    # 4. 重复上述过程，直到将所有噪声去除，得到清晰的视频 latent
    #
    with tqdm(total=len(timesteps), desc="Denoising", leave=False) as pbar:
        for i, t in enumerate(timesteps):
            t = t.unsqueeze(0).to(dtype=pipeline.torch_dtype, device=device)
            
            # Prepare input: concatenate target latents with source latents
            # 0s
            latents_input = torch.cat([latents, source_latents], dim=2)
            
            # Step 1: Predict noise using DIT model with positive prompt
            # 步骤1：使用正向 prompt 通过 DIT 模型预测噪声

            torch.npu.synchronize()
            step1_start = time.time()
            noise_pred_posi = pipeline.dit(
                x=latents_input.to(pipeline.torch_dtype),
                timestep=t,
                context=prompt_embeds,
                cam_emb=cam_emb,
            )
            torch.npu.synchronize()
            step1_time = time.time() - step1_start
            logger.info(f"[Step {i+1}/{len(timesteps)}] Step 1 - DIT positive forward: {step1_time:.2f} s")
            
            # Step 2: Apply CFG (Classifier-Free Guidance) if needed
            # 步骤2：如果需要，应用 CFG（无分类器引导）以提升生成质量
            step2_start = time.time()
            if guidance_scale != 1.0:
                # Compute negative prediction (unconditional prediction)
                # 计算负向预测（无条件预测）
                # 添加同步点，确保之前的操作完成
                torch.npu.synchronize()
                step2a_start = time.time()
                noise_pred_nega = pipeline.dit(
                    x=latents_input.to(pipeline.torch_dtype),
                    timestep=t,
                    context=negative_prompt_embeds,
                    cam_emb=cam_emb,
                )
                torch.npu.synchronize()
                step2a_time = time.time() - step2a_start
                logger.info(f"[Step {i+1}/{len(timesteps)}] Step 2a - DIT negative forward: {step2a_time:.2f} s")
                
                # Apply CFG: noise_pred = unconditional + scale * (conditional - unconditional)
                noise_pred = noise_pred_nega + guidance_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi
            
            # Step 3: Extract only target part (exclude source/condition latents)
            # 0s
            tgt_latent_length = latents.shape[2]
            noise_pred_target = noise_pred[:, :, :tgt_latent_length, ...]
            latents_input_target = latents_input[:, :, :tgt_latent_length, ...]
            
            # Step 4: Update latents using scheduler (remove noise for one step)
            # 0s
            latents = pipeline.scheduler.step(
                noise_pred_target,
                pipeline.scheduler.timesteps[i],
                latents_input_target,
            )
            # Ensure latents are on the correct device after scheduler.step()
            # scheduler.step() may return tensors on a different device, especially after model offloading
            latents = latents.to(device=device)
            
            # Step 5: Compute log probability for GRPO training
            # 步骤5：计算对数概率（用于 GRPO 强化学习训练）
            step5_start = time.time()
            if not hasattr(pipeline.scheduler, 'index_for_timestep'):
                add_index_start = time.time()
                add_index_for_timestep_to_scheduler(pipeline.scheduler)
                add_index_time = time.time() - add_index_start
                logger.info(f"[Step {i+1}/{len(timesteps)}] Step 5a - Add index_for_timestep: {add_index_time:.2f} s")
            
            sde_step_start = time.time()
            _, log_prob, _, _ = sde_step_with_logprob(
                pipeline.scheduler,
                noise_pred_target.float(),
                t,
                latents_input_target.float(),
                prev_sample=latents.float(),
            )
            sde_step_time = time.time() - sde_step_start
            step5_time = time.time() - step5_start
            logger.info(f"[Step {i+1}/{len(timesteps)}] Step 5b - sde_step_with_logprob: {sde_step_time:.2f} s")
            logger.info(f"[Step {i+1}/{len(timesteps)}] Step 5 - Total log prob computation: {step5_time:.2f} s")
            
            # Store latents and log_probs for later use
            # 保存 latents 和 log_probs 供后续使用
            all_latents.append(latents)
            all_log_probs.append(log_prob)
            
            # Update progress bar
            # 更新进度条
            pbar.update(1)
            pbar.set_postfix({
                "timestep": t.item(), 
                "step": f"{i+1}/{len(timesteps)}"
            })
    

    # Decode video (only output videos, not gt_videos)
    pipeline.load_models_to_device(['vae'])
    pipeline.vae.to(device=device)
    videos = pipeline.vae.decode(latents.to(dtype=pipeline.torch_dtype), device=device, **tiler_kwargs)
    pipeline.load_models_to_device([])
    
    videos = videos.to(device=device)
    # gt_videos不再decode和返回
        
    return videos, all_latents, all_log_probs


def eval(pipeline, test_dataloader, test_neg_prompt_embed, config, accelerator, global_step, reward_fn, executor, autocast, num_train_timesteps, ema, transformer_trainable_parameters, epoch):
    if config.train.ema:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

    # 限制评估的样本数量以加快评估速度
    max_eval_samples = getattr(config, 'max_eval_samples', 8)  # 默认评估8个样本
    
    all_rewards = defaultdict(list)
    last_batch_videos = None
    last_batch_gt_videos = None     # <-- Keep gt_videos for saving
    last_batch_prompts = None
    last_batch_rewards = None
    # last_batch_source_videos = None  # <-- Remove source video variable
    
    total_evaluated_samples = 0  # 累计已评估的样本数量

    for test_batch in tqdm(
        test_dataloader,
        desc="Eval: ",
        disable=not accelerator.is_local_main_process,
        position=0,
    ):
        # 如果已经评估了足够的样本，提前退出
        if total_evaluated_samples >= max_eval_samples:
            break
        # Note: the following print statements referenced all_latents before definition, move them later if needed

        # collate_fn now returns (prompts, batch_data, dataset_indices)
        # For eval, we only need prompts and batch_data
        prompts, batch_data, _ = test_batch  # Ignore dataset_indices in eval

        # Extract ReCamMaster data: latents (concatenated target+condition) and camera
        all_latents = batch_data["latents"].to(accelerator.device)  # (batch_size, 16, 42, 60, 104)
        target_cameras = batch_data["camera"].to(accelerator.device)  # (batch_size, 21, 12)

        # Split target (gt) and source (condition) latents for source video decoding (not saved)
        # source_latents = all_latents[:, :, 21:, ...]  # <-- Not used anymore

        # The following block for decoding source videos is now removed

        # Get prompt embeddings from batch_data (already encoded)
        # Remove the second dimension (index 1) to convert (batch_size, 1, seq_len, hidden_dim) -> (batch_size, seq_len, hidden_dim)
        prompt_embeds = batch_data["prompt_emb"]["context"].squeeze(1).to(accelerator.device)  # (batch_size, seq_len, hidden_dim)

        # The last batch may not be full batch_size
        batch_size = all_latents.shape[0]
        
        # 限制当前batch处理的样本数量，确保不超过max_eval_samples
        remaining_samples = max_eval_samples - total_evaluated_samples
        if remaining_samples <= 0:
            break
        
        # 如果当前batch的样本数超过剩余需要的样本数，只处理需要的部分
        actual_batch_size = min(batch_size, remaining_samples)
        if actual_batch_size < batch_size:
            # 只处理前actual_batch_size个样本
            all_latents = all_latents[:actual_batch_size]
            target_cameras = target_cameras[:actual_batch_size]
            prompt_embeds = prompt_embeds[:actual_batch_size]
            prompts = prompts[:actual_batch_size] if isinstance(prompts, list) else prompts
        
        if actual_batch_size < test_neg_prompt_embed.shape[0]:
            current_neg_prompt_embeds = test_neg_prompt_embed[:actual_batch_size]
        else:
            current_neg_prompt_embeds = test_neg_prompt_embed

        # Generate videos using recam_pipeline_with_logprob (same as sampling)
        with autocast():
            with torch.no_grad():
                videos, gt_videos, _, _ = recam_pipeline_with_logprob(
                    pipeline=pipeline,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=current_neg_prompt_embeds,
                    all_latents=all_latents,
                    target_camera=target_cameras,
                    num_inference_steps=config.sample.eval_num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    output_type="tensor",
                    height=config.height,
                    width=config.width,
                    num_frames=config.num_frames,
                )

        # Note: reward_fn should handle video input instead of images
        rewards = executor.submit(reward_fn, videos, gt_videos, prompts)
        # yield to to make sure reward computation starts
        time.sleep(0)
        rewards = rewards.result()
        
        for key, value in rewards.items():
            rewards_gather = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()
            all_rewards[key].append(rewards_gather)

        # 更新累计评估的样本数量
        total_evaluated_samples += actual_batch_size

        # Save last batch for video saving
        last_batch_videos = videos
        last_batch_gt_videos = gt_videos     # <-- Save gt_videos
        last_batch_prompts = prompts
        last_batch_rewards = rewards
        # last_batch_source_videos = source_videos   # <-- Remove

    
    # Handle case where no batches were processed
    if last_batch_videos is None:
        if accelerator.is_main_process:
            logger.warning("No batches processed in eval, skipping video saving")
        if config.train.ema:
            ema.copy_temp_to(transformer_trainable_parameters)
        return

    # 关键修复：在gather之前再次同步，确保所有进程都到达这里
    accelerator.wait_for_everyone()
    
    last_batch_videos_gather = accelerator.gather(torch.as_tensor(last_batch_videos, device=accelerator.device)).float().cpu().numpy()
    # last_batch_source_videos_gather = accelerator.gather(torch.as_tensor(last_batch_source_videos, device=accelerator.device)).float().cpu().numpy()  # <-- Remove
    last_batch_gt_videos_gather = accelerator.gather(torch.as_tensor(last_batch_gt_videos, device=accelerator.device)).float().cpu().numpy()  # <-- Gather gt_videos

    # Gather prompts from all processes
    # Use gather_object for string lists (requires accelerate >= 0.20.0)
    # If not available, fall back to manual collection
    try:
        last_batch_prompts_gather = accelerator.gather_object(last_batch_prompts if last_batch_prompts else [])
        if accelerator.is_main_process:
            # Flatten the list of lists
            last_batch_prompts_gather = [p for prompt_list in last_batch_prompts_gather for p in prompt_list]
            # Ensure length matches videos
            if len(last_batch_prompts_gather) < len(last_batch_videos_gather):
                last_batch_prompts_gather.extend([""] * (len(last_batch_videos_gather) - len(last_batch_prompts_gather)))
            elif len(last_batch_prompts_gather) > len(last_batch_videos_gather):
                last_batch_prompts_gather = last_batch_prompts_gather[:len(last_batch_videos_gather)]
    except (AttributeError, TypeError):
        # Fallback: only use prompts from main process
        if accelerator.is_main_process:
            last_batch_prompts_gather = last_batch_prompts if last_batch_prompts else [""] * len(last_batch_videos_gather)
            if len(last_batch_prompts_gather) < len(last_batch_videos_gather):
                last_batch_prompts_gather.extend([""] * (len(last_batch_videos_gather) - len(last_batch_prompts_gather)))
        else:
            last_batch_prompts_gather = []

    # 关键修复：在最后一个gather之前同步，确保所有进程都完成了前面的gather操作
    accelerator.wait_for_everyone()
    
    last_batch_rewards_gather = {}
    for key, value in last_batch_rewards.items():
        last_batch_rewards_gather[key] = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()

    all_rewards = {key: np.concatenate(value) for key, value in all_rewards.items()}
    if accelerator.is_main_process:
        # Create eval directory for saving videos - use epoch-based naming
        eval_dir = os.path.join(config.logdir if hasattr(config, 'logdir') else "./logs", config.run_name if hasattr(config, 'run_name') else "eval", f"epoch {epoch} eval")
        os.makedirs(eval_dir, exist_ok=True)

        # 限制保存的视频数量为max_eval_samples（默认7个）
        num_samples = min(max_eval_samples, len(last_batch_videos_gather))
        sample_indices = range(num_samples)

        # Convert videos to numpy frames and save as video files
        # Convert numpy array back to torch tensor for tensor2video
        videos_tensor = torch.from_numpy(last_batch_videos_gather).float()
        # source_videos_tensor = torch.from_numpy(last_batch_source_videos_gather).float()  # <-- Remove
        gt_videos_tensor = torch.from_numpy(last_batch_gt_videos_gather).float()   # <-- Add gather for gt_videos

        # Check if values are in [0, 1] range (VAE decode output) or [-1, 1] range
        # tensor2video expects values in [-1, 1] range
        if videos_tensor.min() >= 0 and videos_tensor.max() <= 1:
            # Convert [0, 1] -> [-1, 1]
            videos_tensor = videos_tensor * 2.0 - 1.0
        # if source_videos_tensor.min() >= 0 and source_videos_tensor.max() <= 1:
        #     source_videos_tensor = source_videos_tensor * 2.0 - 1.0  # <-- Remove
        if gt_videos_tensor.min() >= 0 and gt_videos_tensor.max() <= 1:
            gt_videos_tensor = gt_videos_tensor * 2.0 - 1.0   # <-- Range check & normalization

        for idx, index in enumerate(sample_indices):
            video = videos_tensor[index]  # (C, T, H, W)
            # source_video = source_videos_tensor[index]  # (C, T, H, W)   # <-- Remove
            gt_video = gt_videos_tensor[index]  # (C, T, H, W)

            # Use pipeline's tensor2video method to convert to PIL Images
            video_frames = pipeline.tensor2video(video)  # List of PIL Images
            # source_video_frames = pipeline.tensor2video(source_video)  # List of PIL Images  # <-- Remove
            gt_video_frames = pipeline.tensor2video(gt_video)  # List of PIL Images

            # Save output video
            output_video_path = os.path.join(eval_dir, f"output_video_{idx}.mp4")
            save_video(video_frames, output_video_path, fps=20, quality=5)

            # Save gt video
            gt_video_path = os.path.join(eval_dir, f"gt_video_{idx}.mp4")
            save_video(gt_video_frames, gt_video_path, fps=20, quality=5)

        # Save reward information as JSON
        reward_info = []
        for idx, index in enumerate(sample_indices):
            prompt = last_batch_prompts_gather[index] if index < len(last_batch_prompts_gather) else ""
            reward_dict = {k: float(last_batch_rewards_gather[k][index]) for k in last_batch_rewards_gather if index < len(last_batch_rewards_gather[k])}
            reward_info.append({
                "sample_idx": idx,
                "prompt": prompt,
                "rewards": reward_dict,
                "output_video_path": f"output_video_{idx}.mp4",
                # "source_video_path": f"source_video_{idx}.mp4",  # <-- Remove source path from JSON info
                "gt_video_path": f"gt_video_{idx}.mp4",     # Add gt_video path into JSON info
            })

        with open(os.path.join(eval_dir, "reward_info.json"), "w", encoding="utf-8") as f:
            json.dump(reward_info, f, ensure_ascii=False, indent=2)

        # Log reward statistics to wandb (without images)
        for key, value in all_rewards.items():
            print(f"eval_reward_{key}: shape={value.shape}, mean={np.mean(value[value != -10])}")

        wandb.log(
            {
                **{f"eval_reward_{key}": np.mean(value[value != -10]) for key, value in all_rewards.items()},
            },
            step=global_step,
        )

        logger.info(f"Eval videos saved to {eval_dir}")

    if config.train.ema:
        ema.copy_temp_to(transformer_trainable_parameters)

def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

def save_ckpt(save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config):
    if accelerator.is_main_process:
        save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
        save_root_lora = os.path.join(save_root, "lora")
        os.makedirs(save_root_lora, exist_ok=True)
        
        if config.train.ema:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
        
        # Save the model (LoRA weights)
        unwrap_model(transformer, accelerator).save_pretrained(save_root_lora)
        
        if config.train.ema:
            ema.copy_temp_to(transformer_trainable_parameters)
        
        logger.info(f"Checkpoint saved to {save_root_lora}")


def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config

    #############################
    # TODO: 这里需要看看timesteps
    #############################
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    # Set DDP kwargs to handle unused parameters (needed when using CFG with torch.no_grad for negative prompts)
    # 从而解决不知道为什么 projector 和 cam_encoder 没有梯度的问题
    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps
    )

    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="flow-grpo",
            config=config.to_dict(),
            init_kwargs={"wandb": {"name": config.run_name}},
        )
    # logger.info(f"\n{config}")

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)



    # 1. Load Wan2.1 pre-trained models
    # Use accelerator.device to ensure each process uses its own device
    # 使用 accelerator.device 确保每个进程使用自己的设备
    # Note: accelerator.device is available before prepare() and points to the correct device for each process
    model_manager = ModelManager(torch_dtype=torch.bfloat16, device=accelerator.device)
    model_manager.load_models([
        os.path.join(config.pretrained.wan_model, "diffusion_pytorch_model.safetensors"),
        os.path.join(config.pretrained.wan_model, "models_t5_umt5-xxl-enc-bf16.pth"),
        os.path.join(config.pretrained.wan_model, "Wan2.1_VAE.pth"),
    ])
    # Load pipeline on the correct device for each process
    # 在每个进程的正确设备上加载 pipeline
    pipe = WanVideoReCamMasterPipeline.from_model_manager(model_manager, device=accelerator.device)
    
    # Clear model_manager after pipeline creation to free memory
    # 创建 pipeline 后清理 model_manager 以释放内存
    del model_manager
    import gc
    gc.collect()
    torch.npu.empty_cache()


    # 2. Initialize additional modules introduced in ReCamMaster
    # ----- Robust add of cam_encoder and projector: ensure device & dtype match model -----
    # pick a representative device & dtype from the existing model parameters
    rep_param = next(pipe.dit.parameters())
    model_device = rep_param.device
    model_dtype  = rep_param.dtype

    dim = pipe.dit.blocks[0].self_attn.q.weight.shape[0]

    for block in pipe.dit.blocks:
        # create layers and immediately move to correct device & dtype
        # This can solve the following error:
        # # RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one. This error indicates that your module has parameters that were not used in producing loss. You can enable unused parameter detection by passing the keyword argument `find_unused_parameters=True` to `torch.nn.parallel.DistributedDataParallel`, and by 
        # # making sure all `forward` function outputs participate in calculating loss. 
        # # If you already have done the above, then the distributed data parallel module wasn't able to locate the output tensors in the return value of your module's `forward` function. Please include the loss function and the structure of the return value of `forward` of your module when reporting this issue (e.g. list, dict, iterable).
        # # Parameter indices which did not receive grad for rank 0: 8 9 18 19 28 29 38 39 48 49 58 59 68 69 78 79 88 89 98 99 108 109 118 119 128 129 138 139 148 149 158 159 168 169 178 179 188 189 198 199 208 209 218 219 228 229 238 239 248 249 258 259 268 269 278 279 288 289 298 299
        try:
            block.cam_encoder = nn.Linear(12, dim, device=model_device, dtype=model_dtype)
            block.projector   = nn.Linear(dim, dim, device=model_device, dtype=model_dtype)
        except TypeError:
            # older torch: create then move and cast
            block.cam_encoder = nn.Linear(12, dim)
            block.projector   = nn.Linear(dim, dim)
            block.cam_encoder.to(device=model_device, dtype=model_dtype)
            block.projector.to(device=model_device, dtype=model_dtype)

        # initialize weights in-place using proper dtype/device
        with torch.no_grad():
            block.cam_encoder.weight.zero_()
            if block.cam_encoder.bias is not None:
                block.cam_encoder.bias.zero_()
            # set projector weight to identity (use .copy_ to preserve dtype/device)
            eye = torch.eye(dim, device=model_device, dtype=model_dtype)
            block.projector.weight.copy_(eye)
            if block.projector.bias is not None:
                block.projector.bias.zero_()
    # -------------------------------------------------------------------------------------


    # 3. Load ReCamMaster checkpoint
    # Load checkpoint to the correct device for each process
    # 为每个进程加载 checkpoint 到正确的设备
    # 先加载到CPU，避免NPU设备同步冲突
    state_dict = torch.load(config.pretrained.recam_model, map_location='cpu')
    # 同步NPU操作
    torch.npu.synchronize()

    missing, unexpected = pipe.dit.load_state_dict(state_dict, strict=False)
    if missing:
        print("load_state_dict missing keys:", missing)
    if unexpected:
        print("load_state_dict unexpected keys:", unexpected)
    
    # Clear state_dict to free memory
    del state_dict
    gc.collect()
    torch.npu.empty_cache()
        
    # Don't move to device here - accelerator.prepare() will handle device placement
    # pipe.to("npu")  # Removed - let accelerator handle device placement
    # pipe.to(dtype=torch.bfloat16)  # Will be set after prepare

    # 4. set freeze parameters,后面再把要用lora调的解冻
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.dit.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # Move vae and text_encoder to device and cast to inference_dtype
    # Note: For distributed training, it's better to let accelerator.prepare() handle device placement
    # But we need to move non-trainable models here since they won't go through prepare()
    pipe.vae.to(accelerator.device, dtype=inference_dtype)
    pipe.text_encoder.to(accelerator.device, dtype=inference_dtype)
    # pipe.dit will be moved by accelerator.prepare(), but we need to set dtype
    # pipe.dit.to(accelerator.device)  # Removed - let accelerator.prepare() handle this

    # Set correct lora layers
    # "attn.to_q", "attn.to_k", "attn.to_v" → 对应 SelfAttention 中的 q, k, v（Linear 层）
    # "attn.to_out.0" → 对应 SelfAttention 中的 o（输出投影层）
    # "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out" → 这些可能不存在于标准 SelfAttention 中，PEFT 会尝试匹配，找不到则跳过
    # WanModel uses SelfAttention and CrossAttention with modules named q, k, v, o
    # The actual module paths are: blocks.{i}.self_attn.{q,k,v,o} and blocks.{i}.cross_attn.{q,k,v,o}
    # PEFT will match any module ending with these names (q, k, v, o)
    target_modules = [
        "self_attn.q",
        "self_attn.k",
        "self_attn.v",
        "self_attn.o",
        "projector",
        "cam_encoder"
    ]
    # target_modules = [
    #     "self_attn.q",
    #     "self_attn.k",
    #     "self_attn.v",
    #     "self_attn.o",
    #     "projector",
    #     "cam_encoder"
    # ]
    if config.train.lora_path:
        pipe.dit = PeftModel.from_pretrained(pipe.dit, config.train.lora_path)
        # After loading with PeftModel.from_pretrained, all parameters have requires_grad set to False. You need to call set_adapter to enable gradients for the adapter parameters.
        pipe.dit.set_adapter("default")
    else:
        transformer_lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        # get_peft_model() 会为匹配的模块添加 LoRA 适配器，这些适配器参数默认 requires_grad=True
        pipe.dit = get_peft_model(pipe.dit, transformer_lora_config)
        
    # 此处会收集所有 requires_grad=True 的参数（主要是 LoRA 适配器参数），并打印它们的名称
    transformer_trainable_parameters = []
    # print("Trainable parameters (requires_grad=True):")
    for name, param in pipe.dit.named_parameters():
        if param.requires_grad:
            transformer_trainable_parameters.append(param)
            # print(f" - {name}")
    
    num_trainable_params = sum(p.numel() for p in transformer_trainable_parameters)
    logger.info(f"Number of trainable parameter tensors: {len(transformer_trainable_parameters)}")
    logger.info(f"Total number of trainable parameters: {num_trainable_params:,}")

    # This ema setting affects the previous 20 × 8 = 160 steps on average.
    ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=8, device=accelerator.device)
    

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True


    # 5. Initialize the optimizer
    optimizer = torch.optim.AdamW(
        transformer_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # 6.prepare reward fn
    reward_fn = getattr(flow_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)
    eval_reward_fn = getattr(flow_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)


    # 7. prepare train_sampler、train_dataloader、test_dataloader
    # Use ReCamMasterDataset for video + camera data
    # Dataset now loads preprocessed latents from .tensors.pth files
    logger.info("Prepare train dataset...")
    train_dataset = ReCamMasterDataset(
        dataset_path=config.dataset,
        metadata_file_name="metadata-f24-train.csv",
        split='train',
        steps_per_epoch=getattr(config.dataset, 'steps_per_epoch', None)
    )
    test_dataset = ReCamMasterDataset(
        dataset_path=config.dataset,
        metadata_file_name="metadata-f24-test.csv",
        split='test',
        steps_per_epoch=None  # Use full dataset for validation
    )
    # Create an infinite-loop DataLoader
    # 为 GRPO 算法设计的采样器，核心是让每个提示词生成 K 张图片，用于计算组内相对奖励。
    # k=config.sample.k：每个 prompt 生成的图片数量
    # batch_size：每个 GPU 的批次大小
    # num_replicas：GPU 数量
    # 确保 total_samples = num_replicas * batch_size 能被 k 整除
    train_sampler = DistributedKRepeatSampler( 
        dataset=train_dataset,
        batch_size=config.sample.train_batch_size,
        k=config.sample.k,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        seed=24
    )

    # Create a DataLoader; note that shuffling is not needed here because it's controlled by the Sampler.
    # IMPORTANT: num_workers must be 0 to ensure set_epoch() updates are reflected immediately.
    # With num_workers > 0, worker processes create iterators that capture the sampler state at creation time,
    # so set_epoch() changes won't be seen by the workers until the iterator is recreated.
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=0,
        collate_fn=ReCamMasterDataset.collate_fn,
        # persistent_workers=True
    )

    # Create a regular DataLoader
    # 不参与训练，只用于评估，在训练过程中定期运行，计算奖励分数，监控训练进度
    # Note: In distributed training, num_workers should be 0 to avoid deadlock issues
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.sample.test_batch_size,
        collate_fn=ReCamMasterDataset.collate_fn,
        shuffle=False,
        num_workers=0,  # Set to 0 for distributed training to avoid deadlock
    )

    neg_prompt_embed = pipe.encode_prompt("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走", positive=False)["context"][0]
    test_neg_prompt_embed = neg_prompt_embed.repeat(config.sample.test_batch_size, 1, 1)
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.train_batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)

    # initialize stat tracker
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)

    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    
    # Prepare everything with our `accelerator`.
    # This will move pipe.dit to the correct device and wrap it for distributed training
    # Note: In distributed training, all processes must reach this point and synchronize
    transformer, optimizer, train_dataloader, test_dataloader = accelerator.prepare(pipe.dit, optimizer, train_dataloader, test_dataloader)
    
    
    # Set dtype after prepare (accelerator.prepare() may have moved models)
    pipe.dit.to(dtype=torch.bfloat16)
    pipe.vae.to(dtype=torch.bfloat16)
    pipe.text_encoder.to(dtype=torch.bfloat16)
    
    
    # executor to perform callbacks asynchronously. this is beneficial for the llava callbacks which makes a request to a
    # remote server running llava inference.
    executor = futures.ThreadPoolExecutor(max_workers=8)
    # Train!
    samples_per_epoch = (
        config.sample.train_batch_size
        * accelerator.num_processes
        * config.sample.num_batches_per_epoch
    )
    total_train_batch_size = (
        config.train.batch_size
        * accelerator.num_processes
        * config.train.gradient_accumulation_steps
    )
    ##############################################################################################################################################################
    logger.info("***** Running training *****")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    logger.info(f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}")
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")
    # assert config.sample.train_batch_size >= config.train.batch_size
    # assert config.sample.train_batch_size % config.train.batch_size == 0
    # assert samples_per_epoch % total_train_batch_size == 0


    first_epoch = 0
    global_step = 0
    train_iter = iter(train_dataloader)

    for epoch in range(first_epoch, config.num_epochs):
        ################### EVAL ####################
        pipe.dit.eval()
        if epoch % config.eval_freq == 0 and epoch > 0:
            eval(pipe, test_dataloader, test_neg_prompt_embed, config, accelerator, global_step, eval_reward_fn, executor, autocast, num_train_timesteps, ema, transformer_trainable_parameters, epoch)
        if epoch % config.save_freq == 0 and epoch > 0 and accelerator.is_main_process:
            logger.info(f"Saving checkpoint at epoch {epoch}")
            save_ckpt(config.save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config)

        #################### SAMPLING ####################
        # 确保 SAMPLING 阶段所需的模型在正确的设备上
        # VAE 需要在 decode 阶段使用，Text Encoder 可能不需要（prompt 已预编码）
        # 但在某些情况下可能需要重新加载，所以先确保它们在设备上
        pipe.dit.eval()
        
        # 确保 VAE 在设备上（sampling 阶段可能需要 decode）
        pipe.vae.to(accelerator.device)
        # 确保 Text Encoder 在设备上（虽然 prompt 已编码，但为保险起见）
        pipe.text_encoder.to(accelerator.device)
        
        # 清空缓存，为 sampling 阶段做准备
        torch.npu.empty_cache()
        torch.npu.synchronize()
        
        # Prepare scheduler
        pipe.scheduler.set_timesteps(config.sample.num_steps, denoising_strength=1.0, shift=5.0)

        samples = []
        
        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            current_epoch_value = epoch * config.sample.num_batches_per_epoch + i
            train_sampler.set_epoch(current_epoch_value)
            # 重新创建迭代器以确保使用最新的 epoch 值
            # 这是必要的，因为 DataLoader 可能在创建迭代器时缓存了生成器状态
            # 虽然 sampler.__iter__() 每次 yield 时都会重新读取 self.epoch，
            # 但 DataLoader 的迭代器可能缓存了之前的结果
            train_iter = iter(train_dataloader)
            # 从dataloader获取数据，同一epoch内会共享相同的唯一样本集合
            prompts, batch_data, dataset_indices = next(train_iter)
            
            # # 调试信息：打印当前 batch 的详细信息和 prompts
            # if accelerator.is_local_main_process:
            #     print(f"="*100)
            #     print(f"[Batch {i} Debug] Current epoch value: {current_epoch_value}")
            #     print(f"[Batch {i} Debug] Unique prompts: {list(set(prompts))}")
            #     print(f"[Batch {i} Debug] All prompts same? {len(set(prompts)) == 1}")
            #     print(f"="*100)

            # Extract ReCamMaster data: latents (concatenated target+condition) and camera
            all_latents = batch_data["latents"].to(accelerator.device)  # (batch_size, 16, 42, 60, 104)
            # target_latents = batch_data["latents"][:, :, :21, ...].to(accelerator.device)   # (batch_size, 16, 21, 60, 104)
            # source_latents = batch_data["latents"][:, :, 21:, ...].to(accelerator.device)  # (batch_size, 16, 21, 60, 104)
            target_camera = batch_data["camera"].to(accelerator.device)  # (batch_size, 21, 12)
            # Get prompt embeddings from batch_data (already encoded)
            # Remove the second dimension (index 1) to convert (batch_size, 1, seq_len, hidden_dim) -> (batch_size, seq_len, hidden_dim)
            prompt_embeds = batch_data["prompt_emb"]["context"].squeeze(1).to(accelerator.device)

            
            # 缓存文件名可用进程id、epoch和batch索引作区分
            process_index = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
            cache_dir = config.save_dir if hasattr(config, "save_dir") else "./latents_cache"
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, f"latents_cache_rank{process_index}_epoch{epoch}_batch{i}.pth")

            # 2. 优先尝试加载缓存，若存在则返回缓存内容，跳过生成流程（适合在整体for循环开头用）
            # 下面是可用的加载逻辑, 可放在 epoch/i/sample for 头部条件分支处（需结合你主循环使用方式调整）
            use_cache = True  # 设置为True时，优先尝试加载缓存
            cache_loaded = False
            if use_cache and os.path.exists(cache_file):
                logger.info(f"[Rank {process_index}] Loading sample cache from {cache_file}")
                # 先加载到CPU，避免NPU设备同步冲突
                cache_data = torch.load(cache_file, map_location='cpu')
                # 同步NPU操作
                torch.npu.synchronize()
                # 然后移动到目标设备
                videos = cache_data["videos"].to(accelerator.device)
                # 同步NPU操作
                torch.npu.synchronize()
                latent_trajectory = cache_data["latent_trajectory"]
                log_probs = cache_data["log_probs"]
                cache_loaded = True

                # 若已加载，则可选择直接continue后续循环体（或跳过生成流程）
                # 加载第一个视频可视化看看
                # frames = pipe.tensor2video(videos)
                # gt_frames = pipe.tensor2video(gt_videos)
                # # Save video with process-specific filename to avoid overwriting
                # save_video(frames[0], f"test_rank{process_index}_videos.mp4", fps=20, quality=5)
                # save_video(gt_frames[0], f"test_rank{process_index}_gt_videos.mp4", fps=20, quality=5)
            # if False:
            #     iiii=0
            else:
                with autocast():
                    with torch.no_grad():
                        # 生成视频和潜变量轨迹 (ReCamMaster)
                        # videos  (batch_size, Channels=3, Time=81, Height=480, Width=832)
                        videos, latent_trajectory, log_probs = recam_pipeline_with_logprob(
                            pipeline=pipe,
                            prompt_embeds=prompt_embeds,
                            negative_prompt_embeds=sample_neg_prompt_embeds,
                            all_latents=all_latents,
                            target_camera=target_camera,
                            num_inference_steps=config.sample.num_steps,
                            guidance_scale=config.sample.guidance_scale,
                            output_type="tensor",
                            height=config.height,
                            width=config.width,
                            num_frames=config.num_frames
                        )
                
                # 1. 保存 videos、latent_trajectory、log_probs 到缓存文件（每个进程/worker独立的文件名）
                force_save = True
                if force_save or (not os.path.exists(cache_file)):
                    torch.save({
                            "videos": videos,
                            "latent_trajectory": latent_trajectory,
                            "log_probs": log_probs,
                        }, cache_file)
                    logger.info(f"[Rank {process_index}] Saved sample cache to {cache_file}")

            # Stack latents/log_probs so downstream code can slice timestep by timestep
            if isinstance(latent_trajectory, list):
                latent_trajectory = torch.stack(latent_trajectory, dim=1) # (batch_size, num_steps + 1, C, T, H, W)
            if isinstance(log_probs, list):
                log_probs = torch.stack(log_probs, dim=1)   # (batch_size, num_steps)
            
            latent_trajectory = latent_trajectory.to(accelerator.device)
            log_probs = log_probs.to(accelerator.device)
            videos = videos.to(accelerator.device)

            # Repeat scheduler timesteps for each element in the batch
            timesteps = pipe.scheduler.timesteps.repeat(config.sample.train_batch_size, 1)  # (batch_size, num_steps)
            timesteps = timesteps.to(accelerator.device)

            # 保存每个进程、每个batch的所有视频(每个batch有batch_size个视频)，保存到 kankanK 文件夹下
            # 文件名: "epoch{epoch}_rank{process_index}_batch{i}_sample{video_idx}.mp4"
            # 另外生成json文件，标明每个video对应的prompt和相机外参，方便检查

            save_dir = f"logs/kankanK-{epoch}"
            if not os.path.exists(save_dir):
                # 注意：多进程可能并发创建，异常可忽略
                try:
                    os.makedirs(save_dir, exist_ok=True)
                except Exception:
                    pass

            batch_size = videos.shape[0]

            # 保存videos为文件，并收集路径列表
            video_paths = []  # 长度为batch_size的列表，存放各视频的完整保存路径
            video_metadata = []
            
            for video_idx in range(batch_size):
                # 保存生成视频
                frames = pipe.tensor2video(videos[video_idx])
                save_path = os.path.join(save_dir, f"epoch{epoch}_rank{process_index}_batch{i}_sample{video_idx}.mp4")
                save_video(frames, save_path, fps=20, quality=5)
                logger.info(f"[Rank {process_index}] Saved video to {save_path}")
                # 保存完整路径到列表中
                video_paths.append(os.path.abspath(save_path))

                # 记录当前视频的prompt和存储路径
                video_metadata.append({
                    "video_idx": video_idx,
                    "prompt": prompts[video_idx] if prompts is not None and len(prompts) > video_idx else "",
                    "video_path": os.path.abspath(save_path) if os.path.exists(save_path) else "",
                })

            # 保存当前batch的prompt映射JSON文件
            metadata_json_path = os.path.join(save_dir, f"epoch{epoch}_rank{process_index}_batch{i}_metadata.json")
            with open(metadata_json_path, "w", encoding="utf-8") as jf:
                json.dump(video_metadata, jf, indent=2, ensure_ascii=False)
            logger.info(f"[Rank {process_index}] Saved prompt-video mapping to {metadata_json_path}")
            
            # 确保video_paths长度等于batch_size
            assert len(video_paths) == batch_size, f"video_paths length {len(video_paths)} != batch_size {batch_size}"



            
            # print("="*100)
            # print("videos.shape", videos.shape)                           # torch.Size([2, 3, 81, 480, 832])
            # print("gt_videos.shape", gt_videos.shape)                     # torch.Size([2, 3, 81, 480, 832])
            # print("latent_trajectory.shape", latent_trajectory.shape)     # torch.Size([2, 21, 16, 21, 60, 104])
            # print("log_probs.shape", log_probs.shape)                     # torch.Size([2, 20])
            # print("prompt_embeds.shape", prompt_embeds.shape)             # torch.Size([2, 512, 4096])
            # print("all_latents.shape", all_latents.shape)                 # torch.Size([2, 16, 42, 60, 104])
            # print("target_camera.shape", target_camera.shape)             # torch.Size([2, 21, 12])
            # print("timesteps.shape", timesteps.shape)                     # torch.Size([2, 100])
            # print("="*100)


            # 异步奖励计算模式
            # 传入video_paths列表、gt_video=None、prompts和相机外参
            rewards = executor.submit(reward_fn, video_paths, None, prompts, target_camera)
            time.sleep(0)  # Ensure the reward thread starts executing

            # 整理样本，后续在等待 reward 结果后再 collate
            # 保存dataset_indices作为条件标识符，用于后续GRPO分组
            samples.append({
                "prompt_embeds": prompt_embeds,
                "source_latents": all_latents[:, :, 21:, ...],                      # Condition latents
                "target_cameras": target_camera,                                    # (batch_size, 21, 12)
                "timesteps": timesteps,                                             # (sample_batch_size, num_steps)
                "latents": latent_trajectory[:, :-1],                               # (batch_size, num_steps, C, T, H, W)
                "next_latents": latent_trajectory[:, 1:],                           # 下一个状态序列：z_{T-1} 到 z_0
                "log_probs": log_probs,                                             # (batch_size, num_steps) - OLD策略的log_probs
                "rewards": rewards,                                                 # 异步计算的奖励信号
                "dataset_indices": dataset_indices,                                # 条件标识符列表，用于GRPO分组
            })
            
            # 清理当前batch的中间变量
            # videos已经保存为文件，可以立即释放显存
            del videos, latent_trajectory, log_probs, timesteps, all_latents, target_camera, prompt_embeds



        # wait for all rewards to be computed
        # 阶段2：等待所有任务完成（阻塞等待）
        # 如果等待每个奖励计算完成再生成下一批：
        # 需要存储所有中间状态，内存占用会很高
        # 异步模式：
        # 生成完立即释放相关资源，只存储最终的奖励结果
        for sample_idx, sample in enumerate(tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        )):
            rewards = sample["rewards"].result()
            sample["rewards"] = {
                key: torch.as_tensor(value, device=accelerator.device).float()
                for key, value in rewards.items()
            }
            logger.info("sample['rewards']", sample["rewards"])



        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        samples = {
            k: torch.cat([s[k] for s in samples], dim=0)
            if not isinstance(samples[0][k], dict)
            else {
                sub_key: torch.cat([s[k][sub_key] for s in samples], dim=0)
                for sub_key in samples[0][k]
            }
            for k in samples[0].keys()
        }

        # 旧策略留档
        samples["rewards"]["ori_avg"] = samples["rewards"]["avg"]
        # 把 [batch] 形状的平均奖励扩展出时间维，复制成 [batch, num_train_timesteps]，方便未来在每个时间步叠加不同的修正项（ KL 奖励）
        samples["rewards"]["avg"] = samples["rewards"]["avg"].unsqueeze(1).repeat(1, num_train_timesteps)
        
        # 关键修复：在gather之前确保所有进程都完成了samples的整理
        accelerator.wait_for_everyone()
        
        # gather rewards across processes
        gathered_rewards = {key: accelerator.gather(value) for key, value in samples["rewards"].items()}
        gathered_rewards = {key: value.cpu().numpy() for key, value in gathered_rewards.items()}


        


        
        # log rewards and images
        if accelerator.is_main_process:
            log_dict = {"epoch": epoch}
            
            # 记录主要reward（avg和各个reward函数的总分）
            for key, value in gathered_rewards.items():
                if '_strict_accuracy' not in key and '_accuracy' not in key:
                    if isinstance(value, (list, np.ndarray)):
                        log_dict[f"reward_{key}"] = np.mean(value)
                    elif isinstance(value, (int, float)):
                        log_dict[f"reward_{key}"] = value
            
            # 记录详细指标（子指标）
            # 格式：reward_gt_reward_ssim, reward_gt_reward_lpips, reward_optical_reward_temporal_consistency 等
            # 详细指标的键名格式：reward_name_sub_metric_name（例如：gt_reward_ssim, optical_reward_temporal_consistency）
            reward_fn_names = set(config.reward_fn.keys())
            for key, value in gathered_rewards.items():
                if key not in ['avg'] and '_' in key:
                    # 检查是否是详细指标：键名应该以某个reward函数名开头
                    is_detail_metric = False
                    for reward_name in reward_fn_names:
                        if key.startswith(f"{reward_name}_") and key != reward_name:
                            is_detail_metric = True
                            break
                    
                    if is_detail_metric:
                        # 这是一个详细指标
                        if isinstance(value, (list, np.ndarray)):
                            log_dict[f"reward_{key}"] = np.mean(value)
                        elif isinstance(value, (int, float)):
                            log_dict[f"reward_{key}"] = value
            
            wandb.log(log_dict, step=global_step)

        # per-prompt mean/std tracking
        if config.per_prompt_stat_tracking:
            # Use dataset_indices as condition identifiers for GRPO grouping
            # Each unique dataset_index represents a unique condition, and k samples share the same index
            # Gather dataset_indices as tensor across all processes
            gathered_dataset_indices = accelerator.gather(samples["dataset_indices"]).cpu().numpy()
            
            # Ensure length matches gathered_rewards
            if len(gathered_dataset_indices) != len(gathered_rewards['avg']):
                assert False, f"gathered_dataset_indices length ({len(gathered_dataset_indices)}) != gathered_rewards length ({len(gathered_rewards['avg'])}), padding with -1"
                # gathered_dataset_indices = np.concatenate([
                #     gathered_dataset_indices,
                #     np.full(len(gathered_rewards['avg']) - len(gathered_dataset_indices), -1, dtype=gathered_dataset_indices.dtype)
                # ])
            
            # Convert to string identifiers for stat_tracker (it uses string keys internally)
            # This ensures that samples with the same dataset_index are grouped together
            condition_identifiers = [str(int(idx)) for idx in gathered_dataset_indices]

            # if accelerator.is_main_process:
            #     from collections import Counter
            #     print(f"[GRPO Debug] Gathered dataset_indices: {gathered_dataset_indices}")
            #     print(f"[GRPO Debug] Condition identifiers: {condition_identifiers[:10]}...")  # 只打印前10个
            #     print(f"[GRPO Debug] Unique conditions: {len(set(condition_identifiers))}")
            #     print(f"[GRPO Debug] Count per condition: {dict(Counter(condition_identifiers))}")
            
            advantages = stat_tracker.update(condition_identifiers, gathered_rewards['avg'])

            # # 验证同一condition的advantage分组计算
            # if accelerator.is_main_process:
            #     advantages_array = np.array(advantages)
            #     gathered_rewards_avg = gathered_rewards['avg']
                
            #     print(f"[GRPO Debug] Advantages shape: {advantages_array.shape}")
            #     print(f"[GRPO Debug] Gathered rewards avg shape: {gathered_rewards_avg.shape}")
                
            #     # 验证每个condition的advantage分布
            #     print(f"[GRPO Debug] Advantage verification per condition:")
            #     for condition_id in sorted(set(condition_identifiers)):
            #         condition_mask = np.array(condition_identifiers) == condition_id
            #         condition_advantages = advantages_array[condition_mask]
            #         condition_rewards = gathered_rewards_avg[condition_mask]
                    
            #         # 计算组内统计
            #         group_mean = condition_rewards.mean()
            #         group_std = condition_rewards.std() + 1e-4
            #         expected_advantages = (condition_rewards - group_mean) / group_std
                    
            #         print(f"  Condition {condition_id}:")
            #         print(f"    - Samples count: {condition_mask.sum()} (should be {config.sample.k})")
            #         print(f"    - Rewards: {condition_rewards}")
            #         print(f"    - Group mean: {group_mean:.4f}, std: {group_std:.4f}")
            #         print(f"    - Advantages: {condition_advantages[0, :5]}... (first 5 timesteps)")
            #         print(f"    - Expected advantages: {expected_advantages[:5]}... (first 5)")
            #         print(f"    - Advantage mean (should be ~0): {condition_advantages.mean():.4f}")
            #         print(f"    - Advantage std (should be ~1): {condition_advantages.std():.4f}")

            # print("advantages: ", advantages)

            group_size, trained_prompt_num = stat_tracker.get_stats()

            zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(condition_identifiers, gathered_rewards)

            if accelerator.is_main_process:
                wandb.log({
                    "group_size": group_size,
                    "trained_prompt_num": trained_prompt_num,
                    "zero_std_ratio": zero_std_ratio,
                    "reward_std_mean": reward_std_mean,
                },step=global_step)
            stat_tracker.clear()
        else:
            advantages = (gathered_rewards['avg'] - gathered_rewards['avg'].mean()) / (gathered_rewards['avg'].std() + 1e-4)

        # 把 NumPy 的 advantage 转回张量，并依据进程索引切出本地样本，再放回当前设备0
        # 这步操作确保只有当前进程能看到自己的样本对应的本地优势值。
        advantages = torch.as_tensor(advantages, dtype=torch.float32)
        samples["advantages"] = (
            advantages.reshape(accelerator.num_processes, -1, advantages.shape[-1])[accelerator.process_index]
            .to(accelerator.device)
        )
        # if accelerator.is_local_main_process:
        #     print("advantages: ", samples["advantages"].abs().mean())

        # 奖励已经转成 advantages，用不到的直接删，释放显存。
        del samples["rewards"]

        # 把 NumPy 的 advantage 转回张量，并依据进程索引切出本地样本，再放回当前设备。
        mask = (samples["advantages"].abs().sum(dim=1) != 0)
        
        # If the number of True values in mask is not divisible by config.sample.num_batches_per_epoch,
        # randomly change some False values to True to make it divisible
        num_batches = config.sample.num_batches_per_epoch
        true_count = mask.sum()
        if true_count % num_batches != 0:
            false_indices = torch.where(~mask)[0]
            num_to_change = num_batches - (true_count % num_batches)
            if len(false_indices) >= num_to_change:
                random_indices = torch.randperm(len(false_indices))[:num_to_change]
                mask[false_indices[random_indices]] = True
        if accelerator.is_main_process:
            wandb.log({
                "actual_batch_size": mask.sum().item()//config.sample.num_batches_per_epoch,
            }, step=global_step)
        # Filter out samples where the entire time dimension of advantages is zero
        samples = {k: v[mask] for k, v in samples.items()}

        # samples["timesteps"]
        # 采样阶段（调用流水线生成轨迹）时，为每个样本保存的一维时间步序列，形状是 (batch, num_steps)
        # 因此该张量的第 0 维是样本数，第 1 维是单条轨迹包含的扩散/积分时间步数。
        #
        # total_batch_size
        # 过滤后的样本数量（也就是 samples["timesteps"] 的 batch 维度）。在后续会用它来打乱、重新分批这些样本参与训练。
        #
        # num_timesteps
        # 等于 samples["timesteps"] 的第二维，表示每条轨迹包含的时间步个数
        total_batch_size, num_timesteps = samples["timesteps"].shape
        assert num_timesteps == config.sample.num_steps
        
        
        # # 我他妈后面不管了
        # continue

        
        # ========== 显存清理：完全释放SAMPLING阶段使用的模型和变量 ==========
        logger.info(f"[Rank {accelerator.process_index}] Starting memory cleanup after SAMPLING phase...")
        
        # 1. 显式卸载VAE到CPU（训练阶段不需要VAE）
        logger.info("Moving VAE to CPU...")
        pipe.vae.to("cpu")
        
        # 2. 显式卸载Text Encoder到CPU（训练阶段不需要，prompt已经编码）
        logger.info("Moving Text Encoder to CPU...")
        pipe.text_encoder.to("cpu")
        
        # 3. 清理奖励函数使用的模型（multi_score在初始化时就创建了scorer，它们可能仍在GPU上）
        # 注意：multi_score返回的函数是闭包，scorer在score_functions中被创建
        # 由于scorer是通过闭包捕获的，我们无法直接访问它们
        # 但是可以通过强制垃圾回收来清理（如果它们没有被其他引用持有）
        logger.info("Cleaning up reward function models...")
        # 注意：由于multi_score的设计，scorer在每次调用时可能被重新创建（如在clip_v_score中使用try-finally）
        # 但如果scorer被缓存在某个地方，可能需要手动清理
        # 这里我们主要依赖垃圾回收，但也可以尝试清理已知的全局scorer
        try:
            # 如果reward_fn有清理方法，调用它
            if hasattr(reward_fn, 'cleanup') or hasattr(reward_fn, 'clear_cache'):
                if hasattr(reward_fn, 'cleanup'):
                    reward_fn.cleanup()
                if hasattr(reward_fn, 'clear_cache'):
                    reward_fn.clear_cache()
        except Exception as e:
            logger.warning(f"Failed to cleanup reward function: {e}")
        
        # 4. 清理pipeline的缓存状态
        pipe.load_models_to_device([])
        
        # 5. 将samples中的tensors移到CPU以释放GPU显存
        # 注意：samples字典现在保持在内存中，不保存到文件
        def _to_cpu_tree(obj):
            if torch.is_tensor(obj):
                return obj.detach().to("cpu")
            if isinstance(obj, dict):
                return {k: _to_cpu_tree(v) for k, v in obj.items()}
            return obj
        samples = {k: _to_cpu_tree(v) for k, v in samples.items()}
        
        # 6. 强制垃圾回收以释放所有不再使用的对象
        import gc
        gc.collect()
        
        # 7. 清空设备缓存（支持NPU和CUDA）
        torch.npu.empty_cache()
        torch.npu.synchronize()  # 确保所有操作完成
        
        logger.info(f"[Rank {accelerator.process_index}] Memory cleanup completed. Ready for TRAINING phase.")
        # ========== 显存清理结束 ==========



        #################### TRAINING ####################
        # Use samples directly from memory (already moved to CPU during cleanup)
        # Ensure shapes are available
        total_batch_size, num_timesteps = samples["timesteps"].shape


        # num_inner_epochs 控制对同一批采样数据重复训练多少次，以充分利用数据
        for inner_epoch in range(config.train.num_inner_epochs):
            # shuffle samples along batch dimension
            perm = torch.randperm(total_batch_size)
            samples = {k: v[perm] for k, v in samples.items()}

            # rebatch for training
            # torch.randperm 生成随机排列索引，用于打乱数据顺序。对 samples 字典中的所有张量按照相同顺序重新排列
            # 目的：避免模型学习到数据顺序，提高泛化能力
            samples_batched = {
                k: v.reshape(-1, total_batch_size//config.sample.num_batches_per_epoch, *v.shape[1:])
                for k, v in samples.items()
            }

            # Print memory usage of sample tensors before training (only first inner epoch)
            if inner_epoch == 0 and accelerator.is_local_main_process:
                logger.info("\n=== Sample tensors memory usage (CPU, full batch) ===")
                for k, v in samples.items():
                    if torch.is_tensor(v):
                        numel = v.numel()
                        dtype_bytes = v.element_size()
                        total_bytes = numel * dtype_bytes
                        total_gb = total_bytes / (1024**3)
                        logger.info(f"  samples['{k}']: shape={v.shape}, dtype={v.dtype}, size={total_gb:.3f} GB")
                logger.info("=" * 60 + "\n")

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]
            
            # Print memory usage of a single minibatch (only first inner epoch, first minibatch)
            if inner_epoch == 0 and len(samples_batched) > 0 and accelerator.is_local_main_process:
                sample_minibatch = samples_batched[0]
                logger.info("\n=== Single minibatch tensors memory usage (CPU) ===")
                for k, v in sample_minibatch.items():
                    if torch.is_tensor(v):
                        numel = v.numel()
                        dtype_bytes = v.element_size()
                        total_bytes = numel * dtype_bytes
                        total_gb = total_bytes / (1024**3)
                        logger.info(f"  minibatch['{k}']: shape={v.shape}, dtype={v.dtype}, size={total_gb:.3f} GB")
                logger.info("=" * 60 + "\n")


            # train
            pipe.dit.train()
            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                # Move current minibatch tensors to device on-demand to reduce peak memory
                for key, value in list(sample.items()):
                    if torch.is_tensor(value):
                        sample[key] = value.to(device=accelerator.device, non_blocking=True)

                train_timesteps = [step_index  for step_index in range(num_train_timesteps)]
                
                # Prepare source latents for ReCamMaster (already preprocessed)
                source_latents = sample["source_latents"]  # (batch_size, C, T, H, W) - condition latents
                target_cameras = sample["target_cameras"]  # (batch_size, 21, 12)

                
                # Source latents are already encoded, just ensure correct dtype and device
                source_latents = source_latents.to(device=accelerator.device, dtype=pipe.torch_dtype)
                sample["prompt_embeds"] = sample["prompt_embeds"].to(device=accelerator.device)
                sample["timesteps"] = sample["timesteps"].to(device=accelerator.device)
                sample["latents"] = sample["latents"].to(device=accelerator.device)
                sample["next_latents"] = sample["next_latents"].to(device=accelerator.device)
                sample["log_probs"] = sample["log_probs"].to(device=accelerator.device)
                sample["advantages"] = sample["advantages"].to(device=accelerator.device)
                
                # Source latents are already encoded, just ensure correct dtype and device
                source_latents = source_latents.to(dtype=pipe.torch_dtype, device=accelerator.device)
                
                for j in tqdm(
                    train_timesteps,
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.accumulate(transformer):
                        with autocast():
                            prev_sample, log_prob, prev_sample_mean, std_dev_t = compute_log_prob_recam(
                                transformer,
                                pipe,
                                sample,
                                j,
                                sample["prompt_embeds"],
                                target_cameras,
                                source_latents,
                                config,
                                negative_prompt_embeds=train_neg_prompt_embeds[: len(sample["prompt_embeds"])]
                            )
                            if config.train.beta > 0:
                                # KL散度正则化：使用reference模型（基础pretrained模型，通过disable_adapter得到）
                                # 注意：这不是OLD策略，而是用于防止策略偏离基础模型太远的正则化项
                                # OLD策略的log_probs已经在sampling阶段记录在sample["log_probs"]中
                                with torch.no_grad():
                                    with transformer.module.disable_adapter():
                                        _, _, prev_sample_mean_ref, _ = compute_log_prob_recam(
                                            transformer,
                                            pipe,
                                            sample,
                                            j,
                                            sample["prompt_embeds"],
                                            target_cameras,
                                            source_latents,
                                            config,
                                            negative_prompt_embeds=train_neg_prompt_embeds[: len(sample["prompt_embeds"])]
                                        )

                        # # 用于调试报错
                        # # RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one. This error indicates that your module has parameters that were not used in producing loss. You can enable unused parameter detection by passing the keyword argument `find_unused_parameters=True` to `torch.nn.parallel.DistributedDataParallel`, and by 
                        # # making sure all `forward` function outputs participate in calculating loss. 
                        # # If you already have done the above, then the distributed data parallel module wasn't able to locate the output tensors in the return value of your module's `forward` function. Please include the loss function and the structure of the return value of `forward` of your module when reporting this issue (e.g. list, dict, iterable).
                        # # Parameter indices which did not receive grad for rank 0: 8 9 18 19 28 29 38 39 48 49 58 59 68 69 78 79 88 89 98 99 108 109 118 119 128 129 138 139 148 149 158 159 168 169 178 179 188 189 198 199 208 209 218 219 228 229 238 239 248 249 258 259 268 269 278 279 288 289 298 299
                        # # In addition, you can set the environment variable TORCH_DISTRIBUTED_DEBUG to either INFO or DETAIL to print out information about which particular parameters did not receive gradient on this rank as part of this error
                        # for name, param in transformer.named_parameters():
                        #     if param.grad is None:
                        #         print(f"No gradient for Parameter {name}")
                        #     else:
                        #         grad_status = param.requires_grad
                        #         has_grad = param.grad is not None
                        #         print(f"Parameter {name}: requires_grad={grad_status}, has_grad={has_grad}")

                        # grpo logic
                        advantages = torch.clamp(
                            sample["advantages"][:, j],
                            -config.train.adv_clip_max,
                            config.train.adv_clip_max,
                        )
                        ratio = torch.exp(log_prob - sample["log_probs"][:, j])  # new_log_prob - old_log_prob
                        unclipped_loss = -advantages * ratio
                        clipped_loss = -advantages * torch.clamp(
                            ratio,
                            1.0 - config.train.clip_range,
                            1.0 + config.train.clip_range,
                        )
                        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                        if config.train.beta > 0:
                            kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(dim=(1,2,3), keepdim=True) / (2 * std_dev_t ** 2)
                            kl_loss = torch.mean(kl_loss)
                            loss = policy_loss + config.train.beta * kl_loss
                        else:
                            loss = policy_loss

                        info["approx_kl"].append(0.5 * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2))
                        info["clipfrac"].append(torch.mean((torch.abs(ratio - 1.0) > config.train.clip_range).float()))

                        info["policy_loss"].append(policy_loss)
                        if config.train.beta > 0:
                            info["kl_loss"].append(kl_loss)

                        
                        # print("="*100)
                        # print("loss.dtype", loss.dtype)
                        # print("policy_loss.dtype", policy_loss.dtype)
                        # print("kl_loss.dtype", kl_loss.dtype)
                        # print("advantages.dtype", advantages.dtype)
                        # print("log_prob.dtype", log_prob.dtype)
                        # print("sample['log_probs'].dtype", sample["log_probs"].dtype)
                        # print("sample['advantages'].dtype", sample["advantages"].dtype)
                        # print("sample['latents'].dtype", sample["latents"].dtype)
                        # print("="*100)

                        # # ---- begin diagnostic (temporary) ----
                        # try:
                        #     print("DEBUG: loss.dtype =", getattr(loss, "dtype", None))
                        # except Exception:
                        #     pass

                        # # 打印几个 LoRA 相关参数的 dtype（只打印首个匹配，避免刷屏）
                        # cnt = 0
                        # for n, p in transformer.named_parameters():
                        #     if ("lora" in n or "projector" in n or "cam_encoder" in n) and cnt < 8:
                        #         print(f"DEBUG PARAM: {n} dtype={p.dtype} device={p.device} requires_grad={p.requires_grad}")
                        #         cnt += 1
                        #     if cnt >= 8:
                        #         break
                        # # ---- end diagnostic ----


                        loss = loss.to(dtype=torch.bfloat16)
                        info["loss"].append(loss)

                        # backward pass
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(
                                transformer.parameters(), config.train.max_grad_norm
                            )
                        optimizer.step()
                        optimizer.zero_grad()
                        
                        # 清理当前时间步的中间变量
                        del prev_sample, log_prob, prev_sample_mean, std_dev_t
                        if config.train.beta > 0:
                            del prev_sample_mean_ref
                        del advantages, ratio, unclipped_loss, clipped_loss, policy_loss, loss
                        if config.train.beta > 0:
                            del kl_loss

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        # assert (j == train_timesteps[-1]) and (
                        #     i + 1
                        # ) % config.train.gradient_accumulation_steps == 0
                        # log training-related stuff
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        if accelerator.is_main_process:
                            wandb.log(info, step=global_step)
                        global_step += 1
                        info = defaultdict(list)
                if config.train.ema:
                    ema.step(transformer_trainable_parameters, global_step)
                
                # 清理当前 batch 的中间变量，及时释放显存
                del sample
                # 清理训练循环中的中间计算结果
                torch.npu.empty_cache()
            # make sure we did an optimization step at the end of the inner epoch
            # assert accelerator.sync_gradients
            
            # 清理 inner epoch 级别的变量
            del samples_batched, perm
            torch.npu.empty_cache()
        
        # ========== 训练阶段结束后的显存清理 ==========
        logger.info(f"[Rank {accelerator.process_index}] Starting memory cleanup after TRAINING phase...")
        
        # 1. 删除 samples 字典（训练数据已不再需要）
        del samples
        
        # 2. 清理训练过程中的所有中间变量
        # 注意：total_batch_size 和 num_timesteps 在训练阶段开始时定义，这里可以安全删除
        if 'total_batch_size' in locals():
            del total_batch_size
        if 'num_timesteps' in locals():
            del num_timesteps
        
        # 3. 强制垃圾回收
        import gc
        gc.collect()
        
        # 4. 清空设备缓存
        torch.npu.empty_cache()
        torch.npu.synchronize()
        
        logger.info(f"[Rank {accelerator.process_index}] Memory cleanup after TRAINING completed.")
        # ========== 训练阶段显存清理结束 ==========
        
        epoch+=1
        
if __name__ == "__main__":
    app.run(main)

