import math
import torch
from torch import nn
from torch.nn import functional as F
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from model.head_animation.LIA.modules import *

class FaceEncoder(nn.Module):
    def __init__(self, output_channels, size=512):
        super(FaceEncoder, self).__init__()

        channel = [32, 64, 128, 256, 512, 512, 512, output_channels]
        
        self.convs = nn.ModuleList()
        self.convs.append(ConvLayer(3, channel[0], 1))

        in_channel = channel[0]
        for i in range(1, len(channel)):
            out_channel = channel[i]
            self.convs.append(ResBlock(in_channel, out_channel))
            in_channel = out_channel

        self.convs = nn.Sequential(*self.convs)

    def forward(self, x):
        res = []
        h = x
        for conv in self.convs:
            h = conv(h)
            res.append(h)
        
        feats = res[::-1][1:] # from 8x8 to 512x512
        return feats
