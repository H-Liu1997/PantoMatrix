import torch
from torch import nn
from torch.nn import functional as F
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from model.head_animation.LIA.modules import *

class MotionEncoder(nn.Module):
    def __init__(self, latent_dim, size=512):
        super(MotionEncoder, self).__init__()
        
        self.input_size = size
        channel = [32, 64, 128, 256, 512, 512, 512, 512]
        
        self.convs = nn.ModuleList()
        self.convs.append(ConvLayer(3, channel[0], 1))

        in_channel = channel[0]
        for i in range(1, len(channel)):
            out_channel = channel[i]
            self.convs.append(ResBlock(in_channel, out_channel))
            in_channel = out_channel

        self.convs.append(EqualConv2d(in_channel, latent_dim, 4, padding=0, bias=False))
        self.convs = nn.Sequential(*self.convs)

    def forward(self, x):
        res = []
        h = x
        for conv in self.convs:
            h = conv(h)
            res.append(h)
        
        res = res[::-1]
        feats = res[2:] # from 8x8 to 512x512
        latent_code = res[0]
        latent_code = latent_code.view(x.size(0), -1)
        return latent_code, feats



