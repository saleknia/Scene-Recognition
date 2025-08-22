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


        # loaded_data = torch.load('/content/drive/MyDrive/checkpoint/DINOV2_att_SUNAttribute_best.pth', map_location='cuda', weights_only=False)
        # pretrained  = loaded_data['net']
        # model2_dict = self.state_dict()
        # state_dict  = {k:v for k,v in pretrained.items() if ((k in model2_dict.keys()) and (v.shape==model2_dict[k].shape))}

        # model2_dict.update(state_dict)
        # self.load_state_dict(model2_dict)

        # for param in self.model.parameters():
        #     param.requires_grad = False
        
        self.head = nn.Sequential(
                                     nn.Dropout(p=0.5, inplace=True),
                                     nn.Linear(in_features=768, out_features=num_classes, bias=True),
                                )

        # loaded_data = torch.load('/content/drive/MyDrive/checkpoint/DINOV2_att_MIT-67_best.pth', map_location='cuda', weights_only=False)
        # pretrained  = loaded_data['net']
        # model2_dict = self.state_dict()
        # state_dict  = {k:v for k,v in pretrained.items() if ((k in model2_dict.keys()) and (v.shape==model2_dict[k].shape))}

        # model2_dict.update(state_dict)
        # self.load_state_dict(model2_dict)

    def forward(self, x_in):

        features = self.model.forward_features(x_in)['x_norm_patchtokens'] # [B, No, D]
        features = features.mean(dim=1)

        x = self.head(features)

        return x
