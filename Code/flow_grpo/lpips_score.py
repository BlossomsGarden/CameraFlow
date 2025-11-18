
import torch
import torch_npu
import torch.nn.functional as F
from torchvision import models

# ---- VGG-based fallback LPIPS-like (if no lpips lib) ----
class VGGPerceptual(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        vgg_model = models.vgg16(pretrained=False)

        # 加载本地权重
        local_vgg_path = '/home/ma-user/modelarts/user-job-dir/wlh/Model/SeaRaft/vgg16-397923af.pth'
        try:
            state_dict = torch.load(local_vgg_path, map_location='cpu')
            vgg_model.load_state_dict(state_dict)
            # print(f"Successfully loaded VGG16 from local path: {local_vgg_path}")
        except Exception as e:
            # print(f"Failed to load VGG16 from local path {local_vgg_path}: {e}")
            # print("Falling back to pretrained download (if available)")
            # 如果本地加载失败，尝试使用预训练下载（不推荐，但作为fallback）
            vgg_model = models.vgg16(pretrained=True)
        
        vgg = vgg_model.features.eval().to(device)
        for p in vgg.parameters(): p.requires_grad=False
        # use layers to approximate perceptual: relu1_2, relu2_2, relu3_3, relu4_3
        self.layers = vgg
        self.device = device
        
    def forward(self, x, y):
        # x,y: [B,3,H,W] in [0,1], map to [-1,1] as VGG expects unnormalized but works ok
        feats_x = []
        feats_y = []
        xx = x
        yy = y
        feats_idx = [3,8,15,22]  # approximate layers
        out_x = []
        out_y = []
        for i,layer in enumerate(self.layers):
            xx = layer(xx)
            yy = layer(yy)
            if i in feats_idx:
                out_x.append(xx)
                out_y.append(yy)
        loss = 0.0
        for a,b in zip(out_x,out_y):
            loss = loss + F.l1_loss(a,b, reduction='mean')
        return loss
