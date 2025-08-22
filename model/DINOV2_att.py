import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class DINOV2_att(nn.Module):
    def __init__(self, num_classes=67, pretrained=True):
        super(DINOV2_att, self).__init__()

        # Load DINOv2 backbone
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')

        # Freeze all parameters except last transformer block
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.blocks[-1].parameters():
            param.requires_grad = True

        # Feature dimension (ViT-B/14 default)
        feature_dim = 768

        # 102 independent linear heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(feature_dim, 1)
            ) for _ in range(num_classes)
        ])

    def forward(self, x):
        # Extract patch token features
        features = self.backbone.forward_features(x)['x_norm_patchtokens']  # [B, N, D]
        features = features.mean(dim=1)  # Global average pooling over patches -> [B, D]

        # Forward through 102 heads
        logits = [head(features) for head in self.heads]  # list of [B,1]
        logits = torch.cat(logits, dim=1)                 # [B, 102]

        return logits
