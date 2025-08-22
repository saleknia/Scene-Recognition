from re import S, X
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet50, efficientnet_b0, EfficientNet_B0_Weights, efficientnet_b1, EfficientNet_B1_Weights, efficientnet_b2, EfficientNet_B2_Weights, EfficientNet_B3_Weights, efficientnet_b3, EfficientNet_B5_Weights, efficientnet_b4, EfficientNet_B4_Weights, efficientnet_b5, efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, DeepLabV3_MobileNet_V3_Large_Weights
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights
import random
from torchvision.models import resnet50, efficientnet_b4, EfficientNet_B4_Weights
from torch.nn import init
from timm.layers import LayerNorm2d

import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image
import timm
from torchvision.transforms.functional import resize
from torchvision.transforms import FiveCrop, Lambda

from transformers import AutoModelForImageClassification
from .Mobile_netV2 import Mobile_netV2

# base       = Mobile_netV2().cuda()
# checkpoint = torch.load('/content/drive/MyDrive/checkpoint/DINO_base.pth', map_location='cuda')
# base.load_state_dict(checkpoint['net'])

scene      = Mobile_netV2().cuda()
checkpoint = torch.load('/content/drive/MyDrive/checkpoint/DINO_baseline.pth', map_location='cuda')
scene.load_state_dict(checkpoint['net'])
scene = scene.eval()

obj        = Mobile_netV2().cuda()
checkpoint = torch.load('/content/drive/MyDrive/checkpoint/DINOV2_att_MIT-67_best.pth', map_location='cuda')
obj.load_state_dict(checkpoint['net'])
obj = obj.eval()

from .ConvNext import ConvNext
from .ResNet import ResNet

# scene      = ResNet().cuda()
# checkpoint = torch.load('/content/drive/MyDrive/checkpoint/scene.pth', map_location='cuda')
# scene.load_state_dict(checkpoint['net'])

# obj        = ConvNext().cuda()
# checkpoint = torch.load('/content/drive/MyDrive/checkpoint/obj.pth', map_location='cuda')
# obj.load_state_dict(checkpoint['net'])


class Combine(nn.Module):
    def __init__(self, num_classes=67, pretrained=True):
        super(Combine, self).__init__()

        dim = 768

        self.obj_branch   = obj    # distilled from object-based teacher
        self.scene_branch = scene  # distilled from scene-based teacher

        for param in self.parameters():
            param.requires_grad = False

        # self.cross_attn_obj_to_scene = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)
        # self.cross_attn_scene_to_obj = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)
        # self.fusion_fc = nn.Linear(2*dim, dim)  # combine fused features

        self.head = nn.Sequential(
                                    nn.Dropout(p=0.5, inplace=True),
                                    nn.Linear(in_features=dim*2, out_features=num_classes, bias=True),
                                )
                                
    def forward(self, x_in):

        # obj_features   = self.obj_branch.model.forward_features(x_in)
        # scene_features = self.scene_branch.model.forward_features(x_in)

        # obj_tokens   = obj_features['x_norm_patchtokens']    # [B, No, D]
        # scene_tokens = scene_features['x_norm_patchtokens']  # [B, Ns, D]

        # # Cross attention: objects attend to scene
        # obj_with_scene, _ = self.cross_attn_scene_to_obj(obj_tokens, scene_tokens, scene_tokens)
        
        # # Cross attention: scene attends to objects
        # scene_with_obj, _ = self.cross_attn_obj_to_scene(scene_tokens, obj_tokens, obj_tokens)

        # obj_with_scene = obj_tokens   + obj_with_scene
        # scene_with_obj = scene_tokens + scene_with_obj

        # # Pool and fuse
        # obj_feat   = obj_with_scene.mean(dim=1)
        # scene_feat = scene_with_obj.mean(dim=1)

        # obj_feat   = obj_tokens.mean(dim=1)
        # scene_feat = scene_tokens.mean(dim=1)

        # fused_feat = torch.cat([obj_feat, scene_feat], dim=-1)
        # x          = self.head(fused_feat)
        
        o = self.obj_branch(x_in).softmax(dim=1)
        s = self.scene_branch(x_in).softmax(dim=1)
        x = (s + o)
        
        return x

