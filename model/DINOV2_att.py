import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class DINOV2_att(nn.Module):
    def __init__(self, num_classes=67, pretrained=True):
        super(DINOV2_att, self).__init__()

        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.blocks[-1].parameters():
            param.requires_grad = True

        self.head_att = nn.Sequential(
                                    nn.Dropout(p=0.5, inplace=False),
                                    nn.Linear(in_features=768, out_features=num_classes[0], bias=True),
                                )

        self.head_cat = nn.Sequential(
                                    nn.Dropout(p=0.5, inplace=False),
                                    nn.Linear(in_features=768, out_features=num_classes[1], bias=True),
                                )

    def forward(self, x_in):

        features = self.model.forward_features(x_in)['x_norm_patchtokens'] # [B, No, D]
        features = features.mean(dim=1)

        x_att = self.head_att(features)
        x_cat = self.head_cat(features)

        return x_att, x_cat

