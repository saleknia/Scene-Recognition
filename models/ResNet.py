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

from torchvision.transforms import FiveCrop, Lambda

from transformers import AutoModelForImageClassification

class ResNet(nn.Module):
    def __init__(self, num_classes=67, pretrained=True):
        super(ResNet, self).__init__()

        self.model      = models.__dict__['resnet50'](num_classes=365).cuda()
        checkpoint = torch.load('/content/drive/MyDrive/places365_pretrained/resnet50_places365.pth.tar', map_location='cuda')
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        self.model.load_state_dict(state_dict)

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.layer4[-1].parameters():
            param.requires_grad = True

        self.model.fc   = nn.Sequential(
                                    nn.Dropout(p=0.5, inplace=True),
                                    nn.Linear(in_features=2048, out_features=num_classes, bias=True),
                                )

    def forward(self, x_in):

        x = self.model(x_in)
        
        return x
