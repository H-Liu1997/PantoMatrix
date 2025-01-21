import math
import torch
from torch import nn
from torch.nn import functional as F

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from model.head_animation.LIA.motion_encoder import EqualLinear

class Direction(nn.Module):
    def __init__(self, latent_dim, num_direction):
        super(Direction, self).__init__()

        self.weight = nn.Parameter(torch.randn(latent_dim, num_direction))

    def forward(self, input):
        weight = self.weight + 1e-8
        Q, R = torch.qr(weight)  # get eignvector, orthogonal [n1, n2, n3, n4]

        if input is None:
            return Q
        else:
            input_diag = torch.diag_embed(input)  # alpha, diagonal matrix
            out = torch.matmul(input_diag, Q.T)
            out = torch.sum(out, dim=1)
            return out


class FlowEstimator(nn.Module):
    def __init__(self, latent_dim, motion_space=20):
        super(FlowEstimator, self).__init__()

        fc = [EqualLinear(latent_dim, latent_dim)]
        for i in range(3):
            fc.append(EqualLinear(latent_dim, latent_dim))
        fc.append(EqualLinear(latent_dim, motion_space))
        self.fc = nn.Sequential(*fc)

        self.direction = Direction(latent_dim, motion_space)

    def forward(self, ref_fea, tgt_fea):
        feats = self.fc(tgt_fea.view(tgt_fea.size(0), -1))

        ref2tgt_mapping = self.direction(feats)
        tgt_latent = ref_fea + ref2tgt_mapping # reference latent code -> target latent code
        
        return tgt_latent