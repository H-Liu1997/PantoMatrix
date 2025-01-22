import torch
from torch import nn
from diffusers.models.attention_processor import Attention
from .style_modulation import StyleModulation

class MotionGuider(nn.Module):
    def __init__(
        self,
        query_dim: int = 512,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        memory_bank_size: int =64,
        memory_dim: int = 512,
        enable_style_modulation: bool = False,
        latents_dim: int = 512
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim
        self.attn_self = Attention(
            inner_dim,
            memory_dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,    
            # processor=None set the processor if you wanna use more efficient attention operation
        )
        self.attn_other = Attention(
            inner_dim,
            memory_dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,    
            # processor=None set the processor if you wanna use more efficient attention operation
        )
        self.proj_self = nn.Linear(query_dim,inner_dim)
        self.proj_other= nn.Linear(query_dim,inner_dim)
        self.fusion = nn.Sequential(
            nn.Linear(inner_dim,inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim,inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim,query_dim)
        )
        self.memory_self = nn.Parameter(torch.randn(memory_bank_size,memory_dim).unsqueeze(0),requires_grad=True)
        self.memory_other = nn.Parameter(torch.randn(memory_bank_size,memory_dim).unsqueeze(0),requires_grad=True)
        self.enable_style_modulation = enable_style_modulation
        self.style_modulation = None
        if enable_style_modulation:
            self.style_modulation = StyleModulation(latents_dim=latents_dim,memory_dim=memory_dim) 
    
    def set_train_memory(self):
        self.memory_self.requires_grad = True
        self.memory_other.requires_grad = True
    
    def set_fixed_memory(self):
        self.memory_self.requires_grad = False
        self.memory_other.requires_grad = False

    def forward(self,A_self,A_other,ref_latents=None,ref_frames=None):
        assert A_self.shape == A_other.shape
        batch_size, num_frames, query_dim = A_self.shape
        A_self = self.proj_self(A_self)
        A_other = self.proj_other(A_other)
        if ref_latents is not None and self.enable_style_modulation:
            assert batch_size == ref_latents.shape[0]//ref_frames
            memory_self, memory_other = self.style_modulation(self.memory_self,self.memory_other,ref_latents,ref_frames)
        else:
            memory_self = self.memory_self.repeat(batch_size,1,1)
            memory_other = self.memory_other.repeat(batch_size,1,1)
        A_self = self.attn_self(A_self,memory_self)
        A_other = self.attn_other(A_other,memory_other)
        A_fusion = A_self+A_other
        A_fusion = self.fusion(A_fusion)
        return A_fusion
