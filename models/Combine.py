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

# scene      = Mobile_netV2().cuda()
# checkpoint = torch.load('/content/drive/MyDrive/checkpoint/DINO_scene.pth', map_location='cuda')
# scene.load_state_dict(checkpoint['net'])

# obj        = Mobile_netV2().cuda()
# checkpoint = torch.load('/content/drive/MyDrive/checkpoint/DINO_obj.pth', map_location='cuda')
# obj.load_state_dict(checkpoint['net'])

from .ConvNext import ConvNext
from .ResNet import ResNet

scene      = ResNet().cuda()
checkpoint = torch.load('/content/drive/MyDrive/checkpoint/scene.pth', map_location='cuda')
scene.load_state_dict(checkpoint['net'])

obj        = ConvNext().cuda()
checkpoint = torch.load('/content/drive/MyDrive/checkpoint/obj.pth', map_location='cuda')
obj.load_state_dict(checkpoint['net'])


class Combine(nn.Module):
    def __init__(self, num_classes=67, pretrained=True):
        super(Combine, self).__init__()

        self.scene = scene
        self.obj   = obj

    def forward(self, x_in):
        x_in = resize(x_in, (384, 384))
        s = self.scene(x_in).softmax(dim=1)
        # o = self.obj(x_in).softmax(dim=1)
        # x = (s + o)

        return s

