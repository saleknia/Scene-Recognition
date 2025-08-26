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
import torch.nn as nn
from torchvision.transforms import FiveCrop, Lambda

# from efficientvit.cls_model_zoo import create_cls_model
from timm.layers import LayerNorm2d

from transformers import CLIPProcessor, CLIPModel
from PIL import Image

from transformers import AutoModelForImageClassification
from .ConvNext import ConvNext
from .ResNet import ResNet
from .Hybrid import Hybrid

# scene = ResNet().cuda()
# scene = scene.eval()

# checkpoint = torch.load('/content/drive/MyDrive/checkpoint/scene.pth', map_location='cuda')
# scene.load_state_dict(checkpoint['net'])
# scene.model.fc = nn.Identity()

# obj = ConvNext().cuda()
# obj = obj.eval()

# checkpoint = torch.load('/content/drive/MyDrive/checkpoint/obj.pth', map_location='cuda')
# obj.load_state_dict(checkpoint['net'])
# obj.model.head.fc = nn.Identity()

# Hybrid     = Hybrid().cuda()
# checkpoint = torch.load('/content/drive/MyDrive/checkpoint/Hybrid.pth', map_location='cuda')
# Hybrid.load_state_dict(checkpoint['net'])

class Mobile_netV2(nn.Module):
    def __init__(self, num_classes=67, pretrained=True):
        super(Mobile_netV2, self).__init__()

        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.blocks[9:12].parameters():
            param.requires_grad = True

        self.head = nn.Sequential(
                                    nn.Dropout(p=0.5, inplace=True),
                                    nn.Linear(in_features=384, out_features=num_classes, bias=True)
                                )

        # self.scene = ResNet()
        # self.obj   = ConvNext()

        # loaded_data = torch.load('/content/drive/MyDrive/checkpoint/DINO_baseline.pth', map_location='cuda', weights_only=False)
        # pretrained  = loaded_data['net']
        # model_dict  = self.state_dict()
        # state_dict  = {k:v for k,v in pretrained.items() if ((k in model_dict.keys()) and (v.shape==model_dict[k].shape))}
        # model_dict.update(state_dict)
        # self.load_state_dict(model_dict)

    def forward(self, x_in):

        # o = self.obj(x_in).softmax(dim=1)
        # s = scene(x_in).softmax(dim=1)

        # x = self.model(x_in)

        # x = self.head(self.model(x_in))

        features = self.model.forward_features(x_in)['x_norm_patchtokens'] # [B, No, D]
        features = features.mean(dim=1)

        x = self.head(features)

        return x

        # x_t = obj(x_in)

        # if self.training:
        #     return x, x_t
        # else:
        #     return x

        # x_t = scene(x_in)

        # x = self.head(self.model(x_in))

        # if self.training:
        #     return x, x_t
        # else:
        #     return x

        # x_t = Hybrid(x_in)

        # x = self.head(self.model(x_in))

        # if self.training:
        #     return x, x_t
        # else:
        #     return x

