import torch
import torch.nn as nn
from efficientvit.seg_model_zoo import create_efficientvit_seg_model


class seg(nn.Module):
    def __init__(self, num_classes=67, pretrained=True):
        super(seg, self).__init__()

        model = create_efficientvit_seg_model(name="efficientvit-seg-b3-ade20k", pretrained=False)
        model.load_state_dict(torch.load('/content/efficientvit_seg_b3_ade20k.pt')['state_dict'])
        model = model.backbone

        model.input_stem.op_list[0].conv.stride  = (1, 1)
        model.input_stem.op_list[0].conv.padding = (0, 0)

        model.stages[-1].op_list[0].main.depth_conv.conv.stride = (1, 1) 

        self.model = model

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.stages[-1].op_list[-1].parameters():
            param.requires_grad = True

        self.avgpool = nn.AvgPool2d(14, stride=14)
        self.dropout = nn.Dropout(0.5)
        self.head    = nn.Linear(384, num_classes)

    def forward(self, x_in):

        x = self.model(x_in)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.head(x)
        
        return x