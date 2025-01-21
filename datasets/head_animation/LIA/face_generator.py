import math
import torch
from torch import nn
from torch.nn import functional as F
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from model.head_animation.LIA.modules import *

class FaceGenerator(nn.Module):
    def __init__(self, size, latent_dim, blur_kernel=[1, 3, 3, 1], channel_multiplier=1):
        super(FaceGenerator, self).__init__()

        self.size = size
        self.latent_dim = latent_dim
        
        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.input = ConstantInput(self.channels[4]) # 512, 4, 4
        self.conv1 = StyledConv(self.channels[4], self.channels[4], 3, latent_dim, blur_kernel=blur_kernel)
        
        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.to_flows = nn.ModuleList()

        in_channel = self.channels[4]

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]
            # print(i, 2 ** i, in_channel, out_channel)
            # import pdb; pdb.set_trace()
            self.convs.append(StyledConv(in_channel, out_channel, 3, latent_dim, upsample=True, blur_kernel=blur_kernel))
            self.convs.append(StyledConv(out_channel, out_channel, 3, latent_dim, blur_kernel=blur_kernel))
            self.to_rgbs.append(ToRGB(out_channel, latent_dim))

            self.to_flows.append(ToFlow(out_channel, latent_dim))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

    def forward(self, tgt_latent, ref_feats):
        bs = tgt_latent.size(0) 

        inject_index = self.n_latent
        latent = tgt_latent.unsqueeze(1).repeat(1, inject_index, 1)

        out = self.input(latent)
        out = self.conv1(out, latent[:, 0])
        # print('0', out.shape)

        i = 1
        for conv1, conv2, to_rgb, to_flow, feat in zip(self.convs[::2], self.convs[1::2], self.to_rgbs,
                                                       self.to_flows, ref_feats):
            out = conv1(out, latent[:, i])
            out = conv2(out, latent[:, i + 1])
            if out.size(2) == 8:
                out_warp, out, skip_flow = to_flow(out, latent[:, i + 2], feat)
                skip = to_rgb(out_warp)
            else:
                out_warp, out, skip_flow = to_flow(out, latent[:, i + 2], feat, skip_flow)
                skip = to_rgb(out_warp, skip)
            i += 2
            # print(i, out.shape, skip.shape, feat.shape)

        img = skip
        # import pdb; pdb.set_trace()
        return img
