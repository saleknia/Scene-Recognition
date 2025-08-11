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

from .ConvNext import ConvNext
from .ResNet import ResNet

scene      = ResNet().cuda()
checkpoint = torch.load('/content/drive/MyDrive/checkpoint/scene.pth', map_location='cuda')
scene.load_state_dict(checkpoint['net'])

obj        = ConvNext().cuda()
checkpoint = torch.load('/content/drive/MyDrive/checkpoint/obj.pth', map_location='cuda')
obj.load_state_dict(checkpoint['net'])


class Hybrid(nn.Module):
    def __init__(self, num_classes=67, pretrained=True):
        super(Hybrid, self).__init__()

        # self.scene = models.__dict__['resnet50'](num_classes=365).cuda()
        # checkpoint = torch.load('/content/resnet50_places365.pth.tar', map_location='cuda')
        # state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        # self.scene.load_state_dict(state_dict)

        # for param in self.scene.parameters():
        #     param.requires_grad = False

        # for param in self.scene.layer4[-1].parameters():
        #     param.requires_grad = True

        # self.scene.fc = nn.Sequential(
        #                             nn.Dropout(p=0.5, inplace=True),
        #                             nn.Linear(in_features=2048, out_features=256, bias=True)
        #                         )
        # ###########################################
        # ###########################################
        # self.obj = timm.create_model("timm/convnext_tiny.fb_in22k", pretrained=True)

        # for param in self.obj.parameters():
        #     param.requires_grad = False

        # for param in self.obj.stages[-1].blocks[-1].parameters():
        #     param.requires_grad = True

        # self.obj.head.fc = nn.Sequential(
        #                             nn.Dropout(p=0.5, inplace=True),
        #                             nn.Linear(in_features=768, out_features=256, bias=True),
        #                         )

        # for param in self.obj.head.parameters():
        #     param.requires_grad = True
        ###########################################
        ###########################################
    
        self.scene = scene
        self.obj   = obj

        for param in self.obj.parameters():
            param.requires_grad = False

        for param in self.obj.model.head.parameters():
            param.requires_grad = True

        for param in self.scene.parameters():
            param.requires_grad = False

        self.obj.model.head.fc = nn.Sequential(
                                    nn.Dropout(p=0.5, inplace=True),
                                    nn.Linear(in_features=768, out_features=256, bias=True),
                                )


        self.scene.model.fc = nn.Sequential(
                                    nn.Dropout(p=0.5, inplace=True),
                                    nn.Linear(in_features=2048, out_features=256, bias=True)
                                )

        self.head = nn.Sequential(
                            nn.Dropout(p=0.5, inplace=True),
                            nn.Linear(in_features=512, out_features=num_classes, bias=True),
                        )

    def forward(self, x_in):
        
        obj   = self.obj(x_in)
        scene = self.scene(x_in)

        features = torch.cat([obj, scene], dim=1)

        x = self.head(features)
        
        return x
