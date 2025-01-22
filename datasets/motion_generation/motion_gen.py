from diffusers.models.attention import TemporalBasicTransformerBlock, BasicTransformerBlock
from diffusers.models import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from .motion_guider import MotionGuider
from torch import nn
import torch
import time
from typing import Optional
from .motion_gen_utils import TimestepEmbedder, ModulateDiT, modulate, apply_gate
from lightning import LightningModule

class INFPDiffusionTransformerBlock(nn.Module):
    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        latents_dim: int = 32,
        condition_dim: int = 512,
        # num_layers: int = 1,
        cross_attention_dim: Optional[int] = None,
    )-> torch.tensor:
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim
        time_mix_inner_dim = inner_dim
        self.inner_dim = inner_dim
        self.self_attention = TemporalBasicTransformerBlock(
            inner_dim,
            time_mix_inner_dim,
            num_attention_heads,
            attention_head_dim,
        )

        # motion attention
        self.condition_attention = TemporalBasicTransformerBlock(
            inner_dim,
            time_mix_inner_dim,
            num_attention_heads,
            attention_head_dim,
            cross_attention_dim=inner_dim,
        )

        self.temporal_attention = TemporalBasicTransformerBlock(
            inner_dim,
            time_mix_inner_dim,
            num_attention_heads,
            attention_head_dim,
            # cross_attention_dim=cross_attention_dim,
        )
        self.pre_norm = nn.LayerNorm(
            latents_dim, elementwise_affine=False, eps=1e-6
        )
        # self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=latents_dim, eps=1e-6)
        self.proj_in = nn.Linear(latents_dim, inner_dim)
        self.proj_condition = nn.Linear(condition_dim, inner_dim)
        self.proj_out = nn.Linear(inner_dim, latents_dim)
        self.modulation = ModulateDiT(
            latents_dim,
            factor=3,
            act_layer=nn.SiLU
        )

    def forward(
        self,
        hidden_latents,
        condition_latents,
        past_latents,
        vec: torch.Tensor = None, # modulation vector
        num_frames=5,
        n_past_frames=2
    ):
        batch_frames, latents_dim = hidden_latents.shape
        batch_size = batch_frames // num_frames

        # hidden_latents = hidden_latents.reshape(batch_size,num_frames,latents_dim)
        ### 0. timestep modulation
        # mod_shift, mod_scale, mod_gate = self.modulation(vec).chunk(3, dim=-1) # 1, 512
        ### notice that the first dim of mod_shift, mod_scale, mod_gate is batch_size, expand it to batch_frames
        # mod_shift = torch.cat([mod_shift[i].repeat(num_frames,1) for i in range(batch_size)],dim=0)
        # mod_scale = torch.cat([mod_scale[i].repeat(num_frames,1) for i in range(batch_size)],dim=0)
        # mod_gate = torch.cat([mod_gate[i].repeat(num_frames,1) for i in range(batch_size)],dim=0)
        hidden_latents = self.pre_norm(hidden_latents)
        hidden_latents = hidden_latents.reshape(batch_size,num_frames,latents_dim)
        # hidden_latents = modulate(self.pre_norm(hidden_latents), shift=mod_shift, scale=mod_scale)
        # hidden_latents = hidden_latents*(1.0+mod_scale.unsqueeze(1)) + mod_shift.unsqueeze(1)
        hidden_latents = self.proj_in(hidden_latents)
        # batch_size,n_frames,latents_dim
        hidden_latents = hidden_latents.reshape(-1,hidden_latents.shape[-1]).unsqueeze(1)
        #batch_frames,1,latents_dim

        ### 1. self attention
        hidden_latents = self.self_attention(hidden_latents,num_frames=num_frames) # (batch_frames, num_frames, inner_dim)

        ### 2. cross attention
        condition_latents = self.proj_condition(condition_latents)
        hidden_latents = self.condition_attention(hidden_latents,encoder_hidden_states=condition_latents,num_frames=num_frames)

        ### 3. temporal attention
        past_latents = self.proj_in(past_latents)
        past_latents = past_latents.reshape(-1,n_past_frames,past_latents.shape[-1]) # (batch_size,num_frames,inner_dim)
        hidden_latents = self.temporal_attention(hidden_latents,encoder_hidden_states=past_latents,num_frames = num_frames)

        # past_latents=past_latents.permute(0,2,3,1).reshape(-1, height * width, latents_dim)
        # past_latents = self.proj_in(past_latents).reshape(batch_size,n_past_frames,height*width,self.inner_dim)
        # hidden_latents = hidden_latents.reshape(batch_size,num_frames,height*width,self.inner_dim)
        # hidden_latents = torch.cat([past_latents, hidden_latents], dim=1).reshape(-1, height * width, self.inner_dim)
        # hidden_latents = self.temporal_attention(hidden_latents,num_frames=num_frames+n_past_frames) # (batch_frames, height * width, inner_dim)

        hidden_latents = self.proj_out(hidden_latents)
        # hidden_latents = apply_gate(hidden_latents, gate=mod_gate)
        hidden_latents = hidden_latents.permute(0,2,1).squeeze(2)
        return hidden_latents


class INFPDiffusionTransformer(LightningModule):
    # @register_to_config
    def __init__(
        self,
        num_block:int=4,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        latents_dim: int = 512,
        memory_dim: int = 512,
        memory_bank_size: int = 64,
        audio_dim: int = 768,
        enable_style_modulation: bool = False,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            INFPDiffusionTransformerBlock(
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                latents_dim=latents_dim,
                condition_dim=audio_dim
            )     
        for _ in range(num_block)])
        self.time_in = TimestepEmbedder(
            latents_dim,
            nn.SiLU
        )
        self.motion_guider = MotionGuider(
            query_dim=audio_dim,
            memory_bank_size=memory_bank_size,
            memory_dim = memory_dim,
            enable_style_modulation = enable_style_modulation,
            latents_dim=latents_dim
        )
        # self.audio_context_proj = nn.Sequential(
        #     nn.Linear(130,16),
        #     nn.ReLU(),
        #     nn.Linear(16,16),
        #     nn.ReLU(),
        #     nn.Linear(16,1)
        # )

        ### delete it when using 1D AutoEncoder
        # self.ae_proj_in = nn.Sequential(
        #     nn.Linear(512*16*16,512),
        #     nn.ReLU(),
        #     nn.Linear(512,512)
        # )
        
        # self.ae_proj_out = nn.Sequential(
        #     nn.Linear(512,512),
        #     nn.ReLU(),
        #     nn.Linear(512,512*16*16)
        # )

    def audio_preprocess(self,audio_feature):
        # assume that the audio feature is of torch.Size([1, 15, 130, 768])
        B,T,C1,C2 = audio_feature.shape
        assert C1==130
        audio_feature = audio_feature.reshape(B,T,10,13,C2)
        audio_feature = audio_feature[:,:,4:6,-1,:]
        audio_feature = audio_feature.reshape(B,T*2,C2)
        return audio_feature



    def forward(
            self,
            hidden_latents,
            audio_self,
            audio_other,
            past_latents,
            num_frames:int=5,
            n_past_frames:int=2,
            timestep: torch.Tensor = None
    ):
        # Prepare modulation vectors.
        vec = self.time_in(timestep) # timestep modulation
        # print(vec)
        # 
        # print(audio_self.shape)  # torch.Size([1, 15, 130, 768])
        
        # audio_self = self.audio_context_proj(audio_self.permute(0,1,3,2)).squeeze(-1)
        # audio_other = self.audio_context_proj(audio_other.permute(0,1,3,2)).squeeze(-1)
        audio_self = self.audio_preprocess(audio_self)
        audio_other = self.audio_preprocess(audio_other)
        # print(audio_self.shape)

        condition_latents = self.motion_guider(audio_self,audio_other)

        # see the shape:
        # print(f"{hidden_latents.shape=}") # torch.Size([15, 512, 16, 16])
        # print(f"{condition_latents.shape=}") # torch.Size([1, 15, 768])
        # print(f"{past_latents.shape=}") # torch.Size([3, 512, 16, 16])
        ### start to modify the 2D features into 1D
        # BATCH_SIZE,NUM_FRAMES,LATENTS_DIM = hidden_latents.shape
        # hidden_latents = self.ae_proj_in(hidden_latents.flatten(1))
        # past_latents = self.ae_proj_in(past_latents.flatten(1))

        
        ### concat the timestep into latents
        hidden_latents = hidden_latents.reshape(-1,num_frames,hidden_latents.shape[-1])
        hidden_latents = torch.cat([hidden_latents,vec.unsqueeze(1)],dim=1)
        hidden_latents = hidden_latents.reshape(-1,hidden_latents.shape[-1])
        # hidden_latents.shape=torch.Size([15, 512])
        # past_latents.shape=torch.Size([3, 512])
        
        for block in self.blocks:
            hidden_latents = block(hidden_latents,condition_latents,past_latents,vec,num_frames+1,n_past_frames)
        # hidden_latents = self.test_net(hidden_latents)
        hidden_latents = hidden_latents.reshape(-1,num_frames+1,hidden_latents.shape[-1])
        hidden_latents = hidden_latents[:,:num_frames,:].reshape(-1,hidden_latents.shape[-1])
        # hidden_latents = self.ae_proj_out(hidden_latents)
        # hidden_latents=hidden_latents.reshape(-1,LATENTS_DIM,HEIGHT,WIDTH)
        return hidden_latents.unsqueeze(0)
    
