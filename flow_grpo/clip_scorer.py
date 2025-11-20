# Based on https://github.com/RE-N-Y/imscore/blob/main/src/imscore/preference/model.py

from importlib import resources
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from transformers import AutoImageProcessor,CLIPProcessor, CLIPModel
import numpy as np
from PIL import Image



def get_size(size):
    if isinstance(size, int):
        return (size, size)
    elif "height" in size and "width" in size:
        return (size["height"], size["width"])
    elif "shortest_edge" in size:
        return size["shortest_edge"]
    else:
        raise ValueError(f"Invalid size: {size}")
    


def get_image_transform(processor:AutoImageProcessor):
    config = processor.to_dict()
    resize = T.Resize(get_size(config.get("size"))) if config.get("do_resize") else nn.Identity()
    crop = T.CenterCrop(get_size(config.get("crop_size"))) if config.get("do_center_crop") else nn.Identity()
    normalise = T.Normalize(mean=processor.image_mean, std=processor.image_std) if config.get("do_normalize") else nn.Identity()

    return T.Compose([resize, crop, normalise])



class ClipScorer(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device=device
        self.model = CLIPModel.from_pretrained("/home/ma-user/modelarts/user-job-dir/wlh/Model/openai/clip-vit-large-patch14").to(device)
        self.processor = CLIPProcessor.from_pretrained("/home/ma-user/modelarts/user-job-dir/wlh/Model/openai/clip-vit-large-patch14")
        self.tform = get_image_transform(self.processor.image_processor)
        self.eval()
    
    def preprocess_val(self, image):
        """Convert a PIL Image to a tensor for processing. For backward compatibility only."""
        if isinstance(image, Image.Image):
            image_array = np.array(image)
            if len(image_array.shape) == 3:  # HWC
                image_array = image_array.transpose(2, 0, 1)  # CHW
            elif len(image_array.shape) == 2:  # Grayscale
                image_array = np.stack([image_array] * 3, axis=0)  # CHW
            if image_array.dtype == np.uint8:
                image_array = image_array.astype(np.float32) / 255.0
            else:
                image_array = image_array.astype(np.float32)
            return torch.from_numpy(image_array)
        return image

    def _process_tensor(self, pixels):
        """
        Process tensor directly: (B, C, H, W) or (C, H, W) or (B, C, T, H, W)
        pixels: float32 in [0, 1] or uint8 in [0, 255]
        Returns: (B, C, H, W) tensor ready for CLIP
        """
        if not isinstance(pixels, torch.Tensor):
            raise TypeError(f"Expected tensor, got {type(pixels)}")
        
        # Handle (B, C, T, H, W) - reshape to (B*T, C, H, W)
        if len(pixels.shape) == 5:
            B, C, T, H, W = pixels.shape
            pixels = pixels.permute(0, 2, 1, 3, 4).contiguous()  # (B, T, C, H, W)
            pixels = pixels.view(B * T, C, H, W)  # (B*T, C, H, W)
        
        # Ensure (B, C, H, W) format
        if len(pixels.shape) == 3:
            pixels = pixels.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)
        
        # Normalize to [0, 1] if needed
        if pixels.dtype == torch.uint8:
            pixels = pixels.float() / 255.0
        elif pixels.max() > 1.1:
            pixels = pixels / 255.0
        
        # Ensure float32
        pixels = pixels.float()
        
        # Apply transforms (resize, crop, normalize)
        pixels = self.tform(pixels)
        
        return pixels

    def _process(self, pixels):
        """
        Process input: supports tensor (B, C, H, W) or (B, C, T, H, W), PIL Images, or list of PIL Images
        """
        # Handle tensor directly (preferred path)
        if isinstance(pixels, torch.Tensor):
            return self._process_tensor(pixels)
        
        # Handle PIL Images (backward compatibility)
        if isinstance(pixels, Image.Image):
            pixels = self.preprocess_val(pixels).unsqueeze(0)
            return self._process_tensor(pixels)
        
        if isinstance(pixels, list):
            # Convert list of PIL Images to tensor
            tensors = [self.preprocess_val(img) if isinstance(img, Image.Image) else img for img in pixels]
            return self._process_tensor(torch.stack(tensors, dim=0))
        
        raise TypeError(f"Expected tensor, PIL Image, or list, got {type(pixels)}")

    @torch.no_grad()
    def __call__(self, pixels, prompts, return_img_embedding=False):
        # Process pixels first to determine batch size
        pixels_processed = self._process(pixels)
        num_images = pixels_processed.shape[0] if len(pixels_processed.shape) > 0 else 1
        
        # Handle prompts: if single string and multiple images, repeat the prompt
        if isinstance(prompts, str):
            # Single prompt for potentially multiple images
            prompts_list = [prompts] * num_images
        elif isinstance(prompts, (list, tuple)) and len(prompts) == 1 and num_images > 1:
            # Single prompt in a list for multiple images
            prompts_list = list(prompts) * num_images
        else:
            prompts_list = prompts
        
        texts = self.processor(text=prompts_list, padding='max_length', truncation=True, return_tensors="pt").to(self.device)
        pixels_processed = pixels_processed.to(self.device)
        outputs = self.model(pixel_values=pixels_processed, **texts)
        
        # Get logits: shape is [num_images, num_texts]
        logits = outputs.logits_per_image / 30
        # If we have matching number of images and texts, use diagonal
        # Otherwise, take the first column (single prompt case)
        if logits.shape[0] == logits.shape[1]:
            scores = logits.diagonal()
        else:
            scores = logits[:, 0]  # Take first text for all images
        
        if return_img_embedding:
            return scores, outputs.image_embeds
        return scores

    @torch.no_grad()
    def image_similarity(self, pixels, ref_pixels):
        pixels = self._process(pixels).to(self.device)
        ref_pixels = self._process(ref_pixels).to(self.device)

        pixel_embeds = self.model.get_image_features(pixel_values=pixels)
        ref_embeds = self.model.get_image_features(pixel_values=ref_pixels)

        pixel_embeds = pixel_embeds / pixel_embeds.norm(p=2, dim=-1, keepdim=True)
        ref_embeds = ref_embeds / ref_embeds.norm(p=2, dim=-1, keepdim=True)

        sim = pixel_embeds @ ref_embeds.T
        sim = torch.diagonal(sim, 0)
        return sim


def main():
    scorer = ClipScorer(
        device='cuda'
    )

    images=[
    "assets/test.jpg",
    "assets/test.jpg"
    ]
    pil_images = [Image.open(img) for img in images]
    prompts=[
        'an image of cat',
        'not an image of cat'
    ]
    images = [np.array(img) for img in pil_images]
    images = np.array(images)
    images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
    images = torch.tensor(images, dtype=torch.uint8)/255.0
    print(scorer(images, prompts))

if __name__ == "__main__":
    main()