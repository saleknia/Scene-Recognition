import torch
import torch.nn as nn
from .ConvNext import ConvNext

# obj        = ConvNext().cuda()
# checkpoint = torch.load('/content/drive/MyDrive/checkpoint/object.pth', map_location='cuda')
# obj.load_state_dict(checkpoint['net'])

class DINOV3(nn.Module):
    def __init__(self, num_classes=67, pretrained=True):
        super(DINOV3, self).__init__()

        model = torch.hub.load('/content/dinov3', 'dinov3_convnext_tiny', source='local', weights='/content/drive/MyDrive/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth')
        
        self.model = model
        
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.stages[-1].parameters():
            param.requires_grad = True

        for param in self.model.norm.parameters():
            param.requires_grad = True

        for param in self.model.norms.parameters():
            param.requires_grad = True

        self.head = nn.Sequential(
                                    nn.Dropout(p=0.5, inplace=True),
                                    nn.Linear(in_features=768, out_features=num_classes, bias=True),
                                )

    def forward(self, x_in):
    
        x = self.head(self.model(x_in))

        # x_t = obj(x_in)

        # x = self.head(self.model(x_in))

        # if self.training:
        #     return x, x_t
        # else:
        #     return x
        
        return x
