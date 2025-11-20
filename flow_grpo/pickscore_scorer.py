from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np

class PickScoreScorer(torch.nn.Module):
    def __init__(self, device="cuda", dtype=torch.float32):
        super().__init__()
        processor_path = "/home/ma-user/modelarts/user-job-dir/wlh/Model/laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        model_path = "/home/ma-user/modelarts/user-job-dir/wlh/Model/yuvalkirstain/PickScore_v1"
        self.device = device
        self.dtype = dtype
        self.processor = CLIPProcessor.from_pretrained(processor_path)
        self.model = CLIPModel.from_pretrained(model_path).eval().to(device)
        self.model = self.model.to(dtype=dtype)
    
    def _process_tensor(self, images):
        """
        Process tensor images: (B, C, H, W) or (C, H, W)
        images: float32 in [0, 1] or uint8 in [0, 255]
        Returns: list of PIL Images for processor
        """
        if not isinstance(images, torch.Tensor):
            raise TypeError(f"Expected tensor, got {type(images)}")
        
        # Ensure (B, C, H, W) format
        if len(images.shape) == 3:
            images = images.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)
        
        # Normalize to [0, 1] if needed
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        elif images.max() > 1.1:
            images = images / 255.0
        
        # Convert to uint8 numpy array: (B, C, H, W) -> (B, H, W, C)
        images_uint8 = (images * 255).round().clamp(0, 255).to(torch.uint8)
        images_np = images_uint8.cpu().numpy().transpose(0, 2, 3, 1)  # (B, H, W, C)
        
        # Convert to PIL Images
        return [Image.fromarray(img) for img in images_np]
        
    @torch.no_grad()
    def __call__(self, prompt, images):
        # Handle tensor input (preferred path)
        if isinstance(images, torch.Tensor):
            images = self._process_tensor(images)
        
        # Preprocess images
        image_inputs = self.processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}
        
        # Preprocess text
        text_inputs = self.processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        
        # Get embeddings
        image_embs = self.model.get_image_features(**image_inputs)
        image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True)
        
        text_embs = self.model.get_text_features(**text_inputs)
        text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)
        
        # Calculate scores: (num_images, num_texts)
        logit_scale = self.model.logit_scale.exp()
        scores = logit_scale * (text_embs @ image_embs.T)
        
        # Extract diagonal if square matrix, otherwise take first column
        if scores.shape[0] == scores.shape[1]:
            scores = scores.diag()
        else:
            scores = scores[:, 0]
        
        # Normalize to [0, 1]
        scores = scores / 26
        return scores

# Usage example
def main():
    scorer = PickScoreScorer(
        device="cuda",
        dtype=torch.float32
    )
    images=[
    "nasa.jpg",
    ]
    pil_images = [Image.open(img) for img in images]
    prompts=[
        'A astronaut’s glove floating in zero-g with "NASA 2049" on the wrist',
    ]
    print(scorer(prompts, pil_images))

if __name__ == "__main__":
    main()