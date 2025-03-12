import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from transformers import PretrainedConfig

class PeriodicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, period=15, max_seq_len=60): 
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        repeat_num = (max_seq_len // period) + 1
        pe = pe.repeat(1, repeat_num, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    
class TimestepEncoding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        # Fourier embedding
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        self.register_buffer("emb", emb)
        # encoding
        self.encoding = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.Mish(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, t: torch.Tensor):
        """
        :param t: B-dimensional tensor containing timesteps in range [0, 1]
        :return: B x embedding_dim tensor containing timestep encodings
        """
        x = t[:, None] * self.emb[None, :]
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        x = self.encoding(x)
        return x

class FiLM(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.film = nn.Sequential(nn.Mish(), nn.Linear(dim, dim * 2))

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """
        :param x: ... x dim tensor
        :param cond: ... x dim tensor
        :return: ... x dim tensor as scale(cond) * x + bias(cond)
        """
        cond = self.film(cond)
        scale, bias = torch.chunk(cond, chunks=2, dim=-1)
        x = (scale + 1) * x + bias
        return x

class FeedforwardBlock(nn.Module):
    def __init__(self, d_model: int, d_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_feedforward),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_feedforward, d_model),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor):
        """
        :param x: ... x d_model tensor
        :return: ... x d_model tensor
        """
        return self.ff(x)

class SelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None,
    ):
        """
        :param x: B x T x d_model input tensor
        :param attn_mask: B * num_heads x L x S mask with L=target sequence length, S=source sequence length
                          for a float mask: values will be added to attention weight
                          for a binary mask: True indicates that the element is not allowed to attend
        :param key_padding_mask: B x S mask
                          for a float mask: values will be added directly to the corresponding key values
                          for a binary mask: True indicates that the corresponding key value will be ignored
        :return: B x T x d_model output tensor
        """
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        x = self.dropout(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, d_model: int, d_cond: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True,
            kdim=d_cond,
            vdim=d_cond,
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        attn_mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None,
    ):
        """
        :param x: B x T_target x d_model input tensor
        :param cond: B x T_cond x d_cond condition tensor
        :param attn_mask: B * num_heads x L x S mask with L=target sequence length, S=source sequence length
                          for a float mask: values will be added to attention weight
                          for a binary mask: True indicates that the element is not allowed to attend
        :param key_padding_mask: B x S mask
                          for a float mask: values will be added directly to the corresponding key values
                          for a binary mask: True indicates that the corresponding key value will be ignored
        :return: B x T x d_model output tensor
        """
        x = self.cross_attn(
            x,
            cond,
            cond,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        x = self.dropout(x)
        return x

class FilmTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_cond: int,
        num_heads: int,
        d_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = SelfAttention(d_model, num_heads, dropout)
        self.film1 = FiLM(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = CrossAttention(d_model, d_cond, num_heads, dropout)
        self.film2 = FiLM(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.cross_attn_2 = CrossAttention(d_model, d_cond, num_heads, dropout)
        self.film2_2 = FiLM(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.feedforward = FeedforwardBlock(d_model, d_feedforward, dropout)
        self.film3 = FiLM(d_model)

    def forward(
        self,
        x: torch.Tensor,
        cross_cond: torch.Tensor,
        film_cond: torch.Tensor,
        target_mask: torch.Tensor = None,
        target_key_padding_mask: torch.Tensor = None,
        cross_cond_mask: torch.Tensor = None,
        cross_cond_key_padding_mask: torch.Tensor = None,
        cross_cond_2: torch.Tensor = None,
    ):
        """
        :param x: B x T x d_model tensor
        :param cross_cond: B x T x d_cond tensor containing the conditioning input to cross attention layers
        :param film_cond: B x [1 or T] x film_cond tensor containing the conditioning input to FiLM layers
        :return: B x T x d_model tensor
        """
        x1 = self.self_attn(self.norm1(x), target_mask, target_key_padding_mask)
        x = x + self.film1(x1, film_cond)
        x2 = self.cross_attn(
            self.norm2(x), cross_cond, cross_cond_mask, cross_cond_key_padding_mask
        )
        x = x + self.film2(x2, film_cond)
        x2 = self.cross_attn_2(
            self.norm4(x), cross_cond_2, cross_cond_mask, cross_cond_key_padding_mask
        )
        x = x + self.film2_2(x2, film_cond)
        x3 = self.feedforward(self.norm3(x))
        x = x + self.film3(x3, film_cond)
        return x
    

class AutoModelConfig(PretrainedConfig):
    def __init__(self, config_obj=None, **kwargs):
        if config_obj is not None:
            cfg_dict = OmegaConf.to_container(config_obj, resolve=True)
            kwargs.update(cfg_dict)
            self.model_type = kwargs.pop("model_type", "my_model")
        super().__init__(**kwargs)