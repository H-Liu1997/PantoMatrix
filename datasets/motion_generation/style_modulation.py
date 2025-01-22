import torch
from torch import nn

class MotionStyleEncoder(nn.Module):
    def __init__(
        self,
        latents_dim=512
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(latents_dim, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(512, latents_dim, 3, 1, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(latents_dim, latents_dim, 1),
            nn.ReLU(),
            nn.Conv2d(latents_dim, latents_dim, 1),    
        )


    def forward(self,hidden_latents):
        batch_frames, latents_dim, height, width = hidden_latents.shape
        style_embedding = self.encoder(hidden_latents).squeeze()
        return style_embedding

class StyleModulation(nn.Module):
    def __init__(
        self,
        latents_dim=512,
        memory_dim=512,
    ):
        super().__init__()  
        self.projection = nn.Sequential(
            nn.Linear(latents_dim, latents_dim),
            nn.ReLU(),
            nn.Linear(latents_dim, memory_dim)
        )
        self.motion_style_encoder = MotionStyleEncoder(latents_dim)
        
    def forward(
        self,
        memory_self,
        memory_other,
        ref_latents,
        frames=15
    ):
        batch_frames, latents_dim, height, width = ref_latents.shape
        style_embedding = self.motion_style_encoder(ref_latents) # batch_frames, latents_dim
        style_embedding = style_embedding.reshape(batch_frames//frames, frames, latents_dim)
        style_embedding = self.projection(style_embedding)
        # batch, frames, latents_dim
        # temporal mean pooling
        style_embedding = style_embedding.mean(dim=1).squeeze() # batch,latents_dim
        memory_self = memory_self * style_embedding.unsqueeze(1)
        memory_other = memory_other * style_embedding.unsqueeze(1)
        return memory_self, memory_other # batch_size, latents_dim
