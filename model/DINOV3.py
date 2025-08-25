import torch
import torch.nn as nn
from .ConvNext import ConvNext

# obj        = ConvNext().cuda()
# checkpoint = torch.load('/content/drive/MyDrive/checkpoint/object.pth', map_location='cuda')
# obj.load_state_dict(checkpoint['net'])

obj = ConvNext().cuda()
obj = obj.eval()

class DINOV3(nn.Module):
    def __init__(self, num_classes=67, pretrained=True):
        super(DINOV3, self).__init__()

        model = torch.hub.load('/content/dinov3', 'dinov3_convnext_tiny', source='local', weights='/content/drive/MyDrive/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth')
        
        self.downsample_layers = model.downsample_layers
        self.stages            = model.stages

        for param in self.parameters():
            param.requires_grad = False

        for param in self.stages[-1].parameters():
            param.requires_grad = True

        self.head = nn.Sequential(
                                    nn.Dropout(p=0.5, inplace=True),
                                    nn.Linear(in_features=768, out_features=num_classes, bias=True),
                                )

    def forward(self, x):
    
        features_t = obj(x) 

        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        features_s = x

        x_pool = x.mean([-2, -1])  # global average pooling, (N, C, H, W) -> (N, C)

        x = self.head(x_pool)
        
        if self.training:
            return x, (features_s, features_t)
        else:
            return x
