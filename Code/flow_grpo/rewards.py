from PIL import Image
import io
import numpy as np
import torch
import torch_npu
from collections import defaultdict
from tqdm.auto import tqdm

def jpeg_incompressibility():
    def _fn(images, prompts):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        return np.array(sizes), {}

    return _fn


def jpeg_compressibility():
    jpeg_fn = jpeg_incompressibility()

    def _fn(images, prompts):
        rew, meta = jpeg_fn(images, prompts)
        return -rew/500, meta

    return _fn


def aesthetic_score():
    from flow_grpo.aesthetic_scorer import AestheticScorer

    scorer = AestheticScorer(dtype=torch.bfloat16).npu()

    def _fn(images, prompts):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        else:
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)
        scores = scorer(images)
        return scores, {}

    return _fn


def imagereward_score(device):
    from flow_grpo.imagereward_scorer import ImageRewardScorer

    scorer = ImageRewardScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        prompts = [prompt for prompt in prompts]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn

def qwenvl_score(device):
    from flow_grpo.qwenvl import QwenVLScorer

    scorer = QwenVLScorer(dtype=torch.bfloat16, device=device)

    def _fn(images, prompts):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        prompts = [prompt for prompt in prompts]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn


def unifiedreward_score_remote(device):
    """Submits images to UnifiedReward and computes a reward.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 64
    url = "http://10.82.120.15:18085"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        prompts_batched = np.array_split(prompts, np.ceil(len(prompts) / batch_size))

        all_scores = []
        for image_batch, prompt_batch in zip(images_batched, prompts_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "prompts": prompt_batch
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)
            print("response: ", response)
            print("response: ", response.content)
            response_data = pickle.loads(response.content)

            all_scores += response_data["outputs"]

        return all_scores, {}

    return _fn

def unifiedreward_score_sglang(device):
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
        
    async def evaluate_image(prompt, image):
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
        results = await asyncio.gather(*tasks)
        return results

    def _fn(images, prompts):
        # 处理Tensor类型转换
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        
        # 转换为PIL Image并调整尺寸
        images = [Image.fromarray(image).resize((512, 512)) for image in images]

        # 执行异步批量评估
        text_outputs = asyncio.run(evaluate_batch_image(images, prompts))
        score = _extract_scores(text_outputs)
        score = [sc/5.0 for sc in score]
        return score, {}
    
    return _fn


    
def video_ocr_score(device):
    from flow_grpo.ocr import OcrScorer_video_or_image

    scorer = OcrScorer_video_or_image()

    def _fn(images, prompts):
        if isinstance(images, torch.Tensor):
            if images.dim() == 4 and images.shape[1] == 3:
                images = images.permute(0, 2, 3, 1) 
            elif images.dim() == 5 and images.shape[2] == 3:
                images = images.permute(0, 1, 3, 4, 2)
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        scores = scorer(images, prompts)
        # change tensor to list
        return scores, {}

    return _fn


def my_reward():
    """Custom reward function for ReCamMaster training"""
    def _fn(output_video, gt_video, prompts):
        # Return placeholder scores (list of zeros with length = batch_size)
        scores = [0.0] * output_video.shape[0]
        return scores

    return _fn

"""
Dynamic Control————
    temporal_consistency：target video帧间平滑度，接近1好
    visual quality：target <-> GT，cos similarity，接近1好
    Endpoint Error：target <-> GT，EPE，接近0好
"""
def optical_reward(device):
    from flow_grpo.rewards_patch.searaft.optical_reward import optical_eval
    from flow_grpo.rewards_patch.searaft.optical_reward import load_video_frames

    def _fn(output_video, gt_video, prompts):
        # VGG backbone expects float32 tensors; convert from bf16 if needed
        output_video = output_video.float()
        gt_video = gt_video.float()

        batch_size = output_video.shape[0]
        scores = []
        for b in range(batch_size):
            # input (C, T, H, W)
            score = optical_eval(output_video[b], gt_video[b], min_frames=81, device=device, video_layout="cthw")
            scores.append(float(score))
        
        return scores

    return _fn

"""
Camera Control————
    SSIM
    LPIPS
"""
def gt_reward(device):
    import torch.nn.functional as F
    from flow_grpo.lpips_score import VGGPerceptual

    def _fn(output_video, gt_video, prompts):
        # VGG backbone expects float32 tensors and tensors on the same device as the model
        output_video = output_video.float()
        gt_video = gt_video.float()
        # print("="*100)
        # print("output_video.device", output_video.device)
        # print("gt_video.device", gt_video.device)
        # print("device", device)
        # print("="*100)

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

        print("SSIM start")
        print("output_video.device", output_video.device)
        print("gt_video.device", gt_video.device)
        print("device", device)
        vggp = VGGPerceptual(device)
        B, C, T, H, W = output_video.shape
        out_flat = output_video.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)
        gt_flat  = gt_video.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)
        ssim_frame = ssim_torch(out_flat, gt_flat, channel=C)    # (B*T,)
        # Use torch mean directly to avoid numpy dispatch issues on tensors
        mean_ssim = ssim_frame.mean().item()
        print("SSIM end")

        print("LPIPS start")
        # Ensure LPIPS network is on the correct device as well
        lpips_values = []
        for b in range(B):
            for t in range(T):
                lpips = vggp(output_video[b, :, t], gt_video[b, :, t])
                lpips_values.append(lpips.item())

        mean_lpips = float(np.mean(lpips_values))
        print("LPIPS end")
        
        scores = [mean_ssim * 0.3 + mean_lpips * 0.7] * B

        return scores

    return _fn


"""
Video Quality————
    UnifiedReward (TODO)
    PickScore
    CLIP-T
    CLIP-F
    CLIP-V
"""
def clip_score():
    from flow_grpo.clip_scorer import ClipScorer

    scorer = ClipScorer(dtype=torch.float32).npu()

    def _fn(images, prompts):
        if not isinstance(images, torch.Tensor):
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)/255.0
        scores = scorer(images, prompts)
        return scores, {}

    return _fn


def pickscore_score(device):
    from flow_grpo.pickscore_scorer import PickScoreScorer

    scorer = PickScoreScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn


def multi_score(device, score_dict):
    score_functions = {
        "video_ocr": video_ocr_score,
        "imagereward": imagereward_score,
        "pickscore": pickscore_score,
        "qwenvl": qwenvl_score,
        "aesthetic": aesthetic_score,
        "jpeg_compressibility": jpeg_compressibility,
        "unifiedreward": unifiedreward_score_sglang,
        "clipscore": clip_score,
        "my_reward": my_reward,
        "optical_reward": optical_reward,
        "gt_reward": gt_reward,
    }
    score_fns={}
    for score_name, weight in score_dict.items():
        score_fns[score_name] = score_functions[score_name](device) if 'device' in score_functions[score_name].__code__.co_varnames else score_functions[score_name]()

    # During training, only the strict reward is needed, and non-strict rewards don't need to be computed, reducing reward calculation time.
    def _fn(output_video, gt_video, prompts):
        total_scores = []
        score_details = {}

        score_items = list(score_dict.items())
        for score_name, weight in tqdm(
            score_items,
            desc="Evaluating rewards",
            total=len(score_items),
            leave=False,
            dynamic_ncols=True,
        ):
            if score_name in ("optical_reward", "gt_reward"):
                # Move to evaluation device temporarily
                print("当前计算的奖励是：", score_name)
                print("传入的类型output_video.device", output_video.device)
                print("传入的类型gt_video.device", gt_video.device)
                scores = score_fns[score_name](output_video, gt_video, prompts)
            else:
                # CPU-friendly scorers (most convert to numpy/PIL internally)
                scores = score_fns[score_name](output_video, gt_video, prompts)

            score_details[score_name] = scores
            weighted_scores = [weight * score for score in scores]
            
            if not total_scores:
                total_scores = weighted_scores
            else:
                total_scores = [total + weighted for total, weighted in zip(total_scores, weighted_scores)]
        
        score_details['avg'] = total_scores
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