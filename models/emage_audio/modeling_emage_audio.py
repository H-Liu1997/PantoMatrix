import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from transformers import PreTrainedModel
from .configuration_emage_audio import EmageAudioConfig, EmageVQVAEConvConfig, EmageVAEConvConfig
from .processing_emage_audio import Quantizer, VQEncoderV5, VQDecoderV5, WavEncoder, MLP, PeriodicPositionalEncoding, VQEncoderV6, recover_from_mask_ts, rotation_6d_to_axis_angle, velocity2position, axis_angle_to_rotation_6d, rotation_6d_to_matrix, matrix_to_axis_angle, axis_angle_to_matrix, matrix_to_rotation_6d

from torch import Tensor
from torchdiffeq import odeint
from typing import Callable, Optional, Sequence, Tuple, Union
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, BertTokenizer, BertModel, Wav2Vec2Model, Wav2Vec2Config

import inspect
from diffusers.utils.torch_utils import randn_tensor
from diffusers import DiffusionPipeline


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


def audio_to_time_aligned_text_features(inputs, processor, model, tokenizer, bert_model):
    with torch.no_grad():
        logits = model(inputs.input_values).logits  # shape: (1, time_steps, vocab_size)

    predicted_ids_per_timestep = torch.argmax(logits, dim=-1)  # shape: (1, time_steps)
    predicted_ids_per_timestep = predicted_ids_per_timestep[0].cpu().numpy()
    vocab = processor.tokenizer.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}
    tokens_per_timestep = [id_to_token[id] for id in predicted_ids_per_timestep]

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    inputs_bert = tokenizer(transcription, return_tensors="pt")
    input_ids = inputs_bert["input_ids"][0]
    tokens_bert = tokenizer.convert_ids_to_tokens(input_ids)

    with torch.no_grad():
        outputs_bert = bert_model(**inputs_bert.to(inputs.input_values.device))
    all_token_embeddings = outputs_bert.last_hidden_state[0]
    per_timestep_chars = []
    per_timestep_char_indices = []
    for idx, t in enumerate(tokens_per_timestep):
        if t not in ("<pad>", "|"):
            per_timestep_chars.append(t.lower())
            per_timestep_char_indices.append(idx)
    bert_chars = []
    bert_char_indices = []
    for idx, token in enumerate(tokens_bert):
        if token in ("[CLS]", "[SEP]"):
            continue
        token_str = token.replace("##", "")
        for c in token_str:
            bert_chars.append(c)
            bert_char_indices.append(idx)

    s = difflib.SequenceMatcher(None, per_timestep_chars, bert_chars)
    opcodes = s.get_opcodes()
    per_timestep_to_bert_token_idx = {}
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            for k in range(i2 - i1):
                per_timestep_idx = per_timestep_char_indices[i1 + k]
                bert_token_idx = bert_char_indices[j1 + k]
                per_timestep_to_bert_token_idx[per_timestep_idx] = bert_token_idx
    features_per_timestep = []
    check = []
    for i, per_token in enumerate(tokens_per_timestep):
        if i == 0:
            embedding = all_token_embeddings[0]
            check.append("cls")
        elif per_token in ("<pad>", "|"):
            embedding = torch.zeros(all_token_embeddings.shape[-1]).to(inputs.input_values.device)
            check.append(0)
        else:
            if i in per_timestep_to_bert_token_idx:
                bert_idx = per_timestep_to_bert_token_idx[i]
                embedding = all_token_embeddings[bert_idx]
                check.append(tokens_bert[bert_idx])
            else:
                embedding = torch.zeros(all_token_embeddings.shape[-1]).to(inputs.input_values.device)
                check.append(0)
        features_per_timestep.append(embedding)
    features_per_timestep = torch.stack(features_per_timestep)

    updated_check = check.copy()
    for i in range(len(check)):
        if check[i] == 0:
            left = i - 1
            right = i + 1
            left_found = False
            right_found = False

            while left >= 0:
                if check[left] != 0:
                    left_found = True
                    break
                left -= 1

            while right < len(check):
                if check[right] != 0:
                    right_found = True
                    break
                right += 1

            if left_found and right_found:
                if (i - left) <= (right - i):
                    nearest = left
                else:
                    nearest = right
            elif left_found:
                nearest = left
            elif right_found:
                nearest = right
            else:
                continue
            updated_check[i] = updated_check[nearest]
            features_per_timestep[i] = features_per_timestep[nearest]
    features_per_timestep = features_per_timestep.unsqueeze(0)
    return transcription, features_per_timestep, all_token_embeddings


class WrapedWav2Vec(nn.Module):
    def __init__(self, layers=1):
        super(WrapedWav2Vec, self).__init__()
        self.feature_extractor = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").feature_extractor
        self.feature_projection = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").feature_projection
        self.encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").encoder
        self.encoder.layers = self.encoder.layers[:layers]

    def forward(
        self,
        inputs,
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        finetune_audio_low = self.feature_extractor(inputs).transpose(1, 2)
        hidden_states, _ = self.feature_projection(finetune_audio_low.detach())
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = encoder_outputs[0]
        return {"low_level": finetune_audio_low, "high_level": hidden_states}


def inverse_selection_tensor(filtered_t, selection_array, n):
    selection_array = torch.from_numpy(selection_array).cuda()
    original_shape_t = torch.zeros((n, 165)).cuda()
    selected_indices = torch.where(selection_array == 1)[0]
    for i in range(n):
        original_shape_t[i, selected_indices] = filtered_t[i]
    return original_shape_t


class EmageVAEConv(PreTrainedModel):
    config_class = EmageVAEConvConfig
    base_model_prefix = "emage_vaeconv"
    def __init__(self, config):
        super().__init__(config)
        self.encoder = VQEncoderV5(config)
        self.decoder = VQDecoderV5(config)
        
    def forward(self, inputs):
        pre_latent = self.encoder(inputs)
        rec_pose = self.decoder(pre_latent)
        return {
            "rec_pose": rec_pose
            }


class EmageVQVAEConv(PreTrainedModel):
    config_class = EmageVQVAEConvConfig
    base_model_prefix = "emage_vqvaeconv"
    def __init__(self, config):
        super().__init__(config)
        self.encoder = VQEncoderV5(config)
        self.quantizer = Quantizer(config.vae_codebook_size, config.vae_length, config.vae_quantizer_lambda)
        self.decoder = VQDecoderV5(config)
    def forward(self, inputs):
        pre_latent = self.encoder(inputs)
        embedding_loss, vq_latent, _, perplexity = self.quantizer(pre_latent)
        rec_pose = self.decoder(vq_latent)
        return {"poses_feat":vq_latent,"embedding_loss":embedding_loss,"perplexity":perplexity,"rec_pose": rec_pose}
    def map2index(self, inputs):
        pre_latent = self.encoder(inputs)
        index = self.quantizer.map2index(pre_latent)
        return index
    def map2latent(self, inputs):
        pre_latent = self.encoder(inputs)
        index = self.quantizer.map2index(pre_latent)
        z_q = self.quantizer.get_codebook_entry(index)
        return z_q
    def decode(self, index):
        z_q = self.quantizer.get_codebook_entry(index)
        rec_pose = self.decoder(z_q)
        return rec_pose
    def decode_from_latent(self, latent):
        # print(latent.shape)
        z_flattened = latent.contiguous().view(-1, self.quantizer.e_dim)

        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + torch.sum(self.quantizer.embedding.weight**2, dim=1) - 2*torch.matmul(z_flattened, self.quantizer.embedding.weight.t())
        min_encoding_indices = torch.argmin(d, dim=1)
        # print(min_encoding_indices.shape)
        indices = min_encoding_indices.view(latent.shape[0], latent.shape[1])
        z_q = self.quantizer.get_codebook_entry(indices)
        rec_pose = self.decoder(z_q)
        return rec_pose


class EmageVQModel(nn.Module):
    def __init__(self, face_model, upper_model, hands_model, lower_model, global_model):
        super().__init__()
        self.joint_mask_upper = [
          False, False, False, True, False, False, True, False, False, True,
          False, False, True, True, True, True, True, True, True, True,
          True, True, False, False, False, False, False, False, False, False,
          False, False, False, False, False, False, False, False, False, False,
          False, False, False, False, False, False, False, False, False, False,
          False, False, False, False, False
        ]
        self.joint_mask_lower = [
          True, True, True, False, True, True, False, True, True, False,
          True, True, False, False, False, False, False, False, False, False,
          False, False, False, False, False, False, False, False, False, False,
          False, False, False, False, False, False, False, False, False, False,
          False, False, False, False, False, False, False, False, False, False,
          False, False, False, False, False
        ]
        self.vq_model_face = face_model
        self.vq_model_upper = upper_model
        self.vq_model_hands = hands_model
        self.vq_model_lower = lower_model
        self.global_motion = global_model

    def spilt_inputs(self, smplx_body_rot6d, expression, tar_contact=None, tar_trans=None):
        bs, t, j6 = smplx_body_rot6d.shape
        smplx_body_rot6d = smplx_body_rot6d.reshape(bs, t, j6//6, 6)
        jaw_rot6d = smplx_body_rot6d[:, :, 22:23, :].reshape(bs, t, 6)
        face = torch.cat([jaw_rot6d, expression], dim=2)
        upper_rot6d = smplx_body_rot6d[:, :,self.joint_mask_upper, :].reshape(bs, t, 78)
        hands_rot6d = smplx_body_rot6d[:, :,25:55, :].reshape(bs, t, 180)
        lower_rot6d = smplx_body_rot6d[:, :,self.joint_mask_lower, :].reshape(bs, t, 54)
        tar_contact = torch.zeros(bs, t, 4, device=smplx_body_rot6d.device) if tar_contact is None else tar_contact
        tar_trans = torch.zeros(bs, t, 3, device=smplx_body_rot6d.device) if tar_trans is None else tar_trans
        lower = torch.cat([lower_rot6d, tar_trans, tar_contact], dim=2)
        return dict(face=face, upper=upper_rot6d, hands=hands_rot6d, lower=lower)
    
    def map2index(self, smplx_body_rot6d, expression, tar_contact=None, tar_trans=None):
        inputs = self.spilt_inputs(smplx_body_rot6d, expression, tar_contact=tar_contact, tar_trans=tar_trans)
        face_index = self.vq_model_face.map2index(inputs["face"])
        upper_index = self.vq_model_upper.map2index(inputs["upper"])
        hands_index = self.vq_model_hands.map2index(inputs["hands"])
        lower_index = self.vq_model_lower.map2index(inputs["lower"])
        return dict(face=face_index, upper=upper_index, hands=hands_index, lower=lower_index)
    
    def map2latent(self, smplx_body_rot6d, expression, tar_contact=None, tar_trans=None):
        inputs = self.spilt_inputs(smplx_body_rot6d, expression,tar_contact=tar_contact, tar_trans=tar_trans)
        face_latent = self.vq_model_face.map2latent(inputs["face"])
        upper_latent = self.vq_model_upper.map2latent(inputs["upper"])
        hands_latent = self.vq_model_hands.map2latent(inputs["hands"])
        lower_latent = self.vq_model_lower.map2latent(inputs["lower"])
        return dict(face=face_latent, upper=upper_latent, hands=hands_latent, lower=lower_latent)
    
    def decode(self, face_index=None, upper_index=None, hands_index=None, lower_index=None, 
               face_latent=None, upper_latent=None, hands_latent=None, lower_latent=None, 
            get_global_motion=False, ref_trans=None):
        
        for input_tensor in [face_index, upper_index, hands_index, lower_index, face_latent, upper_latent, hands_latent, lower_latent]:
            if input_tensor is not None:
                bs, t = input_tensor.shape[:2]
                break
  
        if face_index is not None:
            face_mix = self.vq_model_face.decode(face_index) # bs, t, 106
            face_jaw_6d, expression = face_mix[:, :, :6], face_mix[:, :, 6:]
            face_jaw = rotation_6d_to_axis_angle(face_jaw_6d)
        elif face_latent is not None:
            face_mix = self.vq_model_face.decode_from_latent(face_latent)
            face_jaw_6d, expression = face_mix[:, :, :6], face_mix[:, :, 6:]
            face_jaw = rotation_6d_to_axis_angle(face_jaw_6d)
        else:
            face_jaw = torch.zeros(bs, t, 3, device=self.vq_model_face.device)
            expression = torch.zeros(bs, t, 100, device=self.vq_model_face.device)

        if upper_index is not None:
            # print(upper_index)
            upper_6d = self.vq_model_upper.decode(upper_index) # bs, t, 78
            upper = rotation_6d_to_axis_angle(upper_6d.reshape(bs, t, -1, 6)).reshape(bs, t, -1)
        elif upper_latent is not None:
            upper_6d = self.vq_model_upper.decode_from_latent(upper_latent)
            upper = rotation_6d_to_axis_angle(upper_6d.reshape(bs, t, -1, 6)).reshape(bs, t, -1)
        else:
            upper = torch.zeros(bs, t, 39, device=self.vq_model_upper.device)

        if hands_index is not None:
            hands_6d = self.vq_model_hands.decode(hands_index)
            hands = rotation_6d_to_axis_angle(hands_6d.reshape(bs, t, -1, 6)).reshape(bs, t, -1)
        elif hands_latent is not None:
            hands_6d = self.vq_model_hands.decode_from_latent(hands_latent)
            hands = rotation_6d_to_axis_angle(hands_6d.reshape(bs, t, -1, 6)).reshape(bs, t, -1)
        else:
            hands = torch.zeros(bs, t, 90, device=self.vq_model_hands.device)
        
        if lower_index is not None:
            lower_mix = self.vq_model_lower.decode(lower_index)
            lower_6d, transfoot = lower_mix[:, :, :-7], lower_mix[:, :, -7:]
            lower = rotation_6d_to_axis_angle(lower_6d.reshape(bs, t, -1, 6)).reshape(bs, t, -1)
        elif lower_latent is not None:
            lower_mix = self.vq_model_lower.decode_from_latent(lower_latent)
            lower_6d, transfoot = lower_mix[:, :, :-7], lower_mix[:, :, -7:]
            lower = rotation_6d_to_axis_angle(lower_6d.reshape(bs, t, -1, 6)).reshape(bs, t, -1)
        else:
            lower = torch.zeros(bs, t, 27, device=self.vq_model_lower.device)
            transfoot = torch.zeros(bs, t, 7, device=self.vq_model_lower.device)
            lower_6d = axis_angle_to_rotation_6d(lower.reshape(bs, t, -1, 3)).reshape(bs, t, -1)
            lower_mix = torch.cat([lower_6d, transfoot], dim=-1)

        upper2all = recover_from_mask_ts(upper, self.joint_mask_upper)
        hands2all = recover_from_mask_ts(hands, [False]*25+[True]*30)
        lower2all = recover_from_mask_ts(lower, self.joint_mask_lower)
        
        all_motion_axis_angle = upper2all + hands2all + lower2all
        all_motion_axis_angle[:, :, 22*3:22*3+3] = face_jaw
        all_motion_rot6d = axis_angle_to_rotation_6d(all_motion_axis_angle.reshape(bs, t, 55, 3)).reshape(bs, t, 55*6)

        all_motion4inference = torch.cat([all_motion_rot6d, transfoot], dim=2) # 330 + 3 + 4
        
        global_motion = None
        if get_global_motion:
            global_motion = self.get_global_motion(lower_mix, ref_trans)
        return dict(expression=expression, all_motion4inference=all_motion4inference, motion_axis_angle=all_motion_axis_angle, trans=global_motion)
    
    def get_global_motion(self, lower_body, ref_trans):
        global_motion = self.global_motion(lower_body)
        rec_trans_v_s = global_motion["rec_pose"][:, :, 54:57]
        if len(ref_trans.shape) == 2:
            ref_trans = ref_trans.unsqueeze(0).repeat(rec_trans_v_s.shape[0], 1, 1)
        
        rec_x_trans = velocity2position(rec_trans_v_s[:, :, 0:1], 1/30, ref_trans[:, 0, 0:1])
        rec_z_trans = velocity2position(rec_trans_v_s[:, :, 2:3], 1/30, ref_trans[:, 0, 2:3])
        rec_y_trans = rec_trans_v_s[:,:,1:2]
        global_motion = torch.cat([rec_x_trans, rec_y_trans, rec_z_trans], dim=-1)
        return global_motion
    

class Pose2PosePipeline(DiffusionPipeline):
    _optional_components = []
    def __init__(
        self,
        model,
        scheduler=None,
    ):
        super().__init__()
        self.register_modules(
            model=model,
        )
        if scheduler is not None:
            self.setup_scheduler(scheduler)
    
    def setup_scheduler(self, scheduler):
        self.register_modules(scheduler=scheduler)
        
    def prepare_extra_step_kwargs(self, generator, eta):
        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs
    
    @torch.no_grad()
    def __call__(
        self,
        num_inference_steps,
        device,
        generator,
        eta=0.0,
        callback=None,
        callback_steps=1,
        **model_extras):
        dtype = model_extras["masked_motion"].dtype
        bs, n, _ = model_extras["masked_motion"].shape
        
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        latents = randn_tensor(
            model_extras["masked_motion"].shape, generator=generator, device=device, dtype=dtype
        )
        latents = latents * self.scheduler.init_noise_sigma
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Create a batch of timesteps
                t_batch = torch.full((bs,), t, device=device, dtype=torch.long)
                latent_model_input = self.scheduler.scale_model_input(
                    latents, t
                )
                noise_pred = self.model(
                    x=latent_model_input, t=t_batch, audio=model_extras["audio"], 
                    speaker_id=model_extras["speaker_id"], masked_motion=model_extras["masked_motion"], mask=model_extras["mask"],
                    use_audio=True)
                # Compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]
                # Call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)
        face_latent = latents
        return face_latent
    
    
class EmageAudioModel(PreTrainedModel):
    config_class = EmageAudioConfig
    base_model_prefix = "emage_audio"
    def __init__(self, config: EmageAudioConfig):
        super().__init__(config)
        self.cfg = config
        # audio encoder
        # self.audio_encoder_face = WavEncoder(self.cfg.audio_f)
        self.audio_encoder_face = WrapedWav2Vec(layers=4)        
        self.audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        # speaker id
        self.speaker_embedding_face = nn.Embedding(self.cfg.speaker_dims, self.cfg.hidden_size)
        # mask embedding
        self.mask_embedding = nn.Parameter(torch.zeros(1,1,self.cfg.vae_codebook_size))
        nn.init.normal_(self.mask_embedding, 0, self.cfg.hidden_size**-0.5)
        
        # motion memory
        # self.memory_bank = nn.Parameter(torch.zeros(1, 64, self.cfg.hidden_size))
        # nn.init.normal_(self.memory_bank, 0, self.cfg.hidden_size**-0.5)
        
        self.position_embeddings = PeriodicPositionalEncoding(self.cfg.hidden_size, period=self.cfg.pose_length, max_seq_len=self.cfg.pose_length)
        # self.audio_motion_cross_attn_layer = nn.TransformerDecoderLayer(d_model=self.cfg.hidden_size,nhead=4,dim_feedforward=self.cfg.hidden_size*2)
        # face decoder
        self.input_up = nn.Linear(self.cfg.vae_codebook_size, self.cfg.hidden_size)
        self.input_up_2 = nn.Linear(self.cfg.vae_codebook_size, self.cfg.hidden_size)
        self.audio_face_motion_proj = nn.Linear(self.cfg.hidden_size, self.cfg.hidden_size)
        # self.face_motion_cross_audio = nn.TransformerDecoder(self.audio_motion_cross_attn_layer, num_layers=4)
        self.face_motion_cross_audio = nn.ModuleList(
            [
                FilmTransformerDecoderLayer(
                    self.cfg.hidden_size, self.cfg.hidden_size, 4, self.cfg.hidden_size*2, 0.1
                )
                for _ in range(4)
            ]
        ) 
        self.face_out_proj = nn.Linear(self.cfg.hidden_size, self.cfg.vae_codebook_size)
        self.face_cls = MLP(self.cfg.vae_codebook_size, self.cfg.hidden_size, self.cfg.vae_codebook_size)
        self.time_embed = TimestepEncoding(self.cfg.hidden_size)
        
        self.inference_pipeline = Pose2PosePipeline(model=self)
        self.cross_attn = CrossAttention(self.cfg.hidden_size, self.cfg.hidden_size, 1, 0.1)
        self.norm = nn.LayerNorm(self.cfg.hidden_size)
        
        self.in_up_3 = nn.Linear(self.cfg.vae_codebook_size, self.cfg.hidden_size)
        self.style_encoder_layer = nn.TransformerEncoderLayer(d_model=self.cfg.hidden_size, nhead=4, dim_feedforward=self.cfg.hidden_size*2)
        self.style_encoder = nn.TransformerEncoder(self.style_encoder_layer, num_layers=2)
        self.film_style = FiLM(self.cfg.hidden_size//2)
        
        self.memory_base = nn.Parameter(torch.zeros(1, 64, self.cfg.hidden_size//2))
        self.memory_style = nn.Parameter(torch.zeros(1, 64, self.cfg.hidden_size//2))
        nn.init.kaiming_normal_(self.memory_base, mode='fan_out', nonlinearity='relu')
        nn.init.uniform_(self.memory_style, -0.1, 0.1) 
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.cfg.hidden_size, self.cfg.hidden_size*2),
            nn.GELU(),
            nn.Linear(self.cfg.hidden_size*2, self.cfg.hidden_size)
        )
        self.style_down = nn.Linear(self.cfg.hidden_size, self.cfg.hidden_size//2)
        
    def forward(self, x, t, audio=None, speaker_id=None, masked_motion=None, mask=None, use_audio=True):
        # mask motion
        # masked_embeddings = self.mask_embedding.expand_as(masked_motion) # bs, n, d
        style_motion = masked_motion
        masked_motion = torch.where(mask==1, 0.0, masked_motion) # frist 4 is gt
        
        audio_list = [i.cpu().numpy() for i in audio]
        inputs = self.audio_processor(audio_list, sampling_rate=16000, return_tensors="pt", padding=True).to(audio.device)
        audio2face_fea = self.audio_encoder_face(inputs.input_values)["high_level"]
        audio2face_fea = F.interpolate(audio2face_fea.transpose(1, 2), scale_factor=245 / 400, mode="linear", align_corners=True).transpose(1, 2)
        bs, n, _ = x.shape
        if audio2face_fea.shape[1] > n:
          audio2face_fea = audio2face_fea[:, :n]
        masked_motion = masked_motion[:, :n]
        style_motion = style_motion[:, :n]
        
        # style encoder
        style_motion = self.in_up_3(style_motion)
        style_motion = self.position_embeddings(style_motion)
        style_motion = self.style_encoder(style_motion)
        style_vec = style_motion[:, 0:1] # bs, 1, d
        style_vec = self.style_down(style_vec)
        # print(masked_motion.shape, x.shape)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        # print(t)      
        emb = self.time_embed(t).unsqueeze(1).repeat(1,n,1)
        # print(emb.shape, audio2face_fea.shape)
        # speaker_face_fea_proj = self.speaker_embedding_face(speaker_id)
        # x = torch.cat([x, masked_motion], dim=2)
        masked_motion = self.input_up_2(masked_motion[:,:self.cfg.seed_frames])
        x = self.input_up(x)
        x = self.position_embeddings(x)
        
        # assume audio_feature is normalized
        audio2face_fea_proj = self.audio_face_motion_proj(audio2face_fea)
        audio2face_fea_proj = self.position_embeddings(audio2face_fea_proj)
        base_memory = self.memory_base.repeat(bs, 1, 1)  # [B,64,d/2]
        # print(self.memory_style.repeat(bs,1,1).shape, style_vec.repeat(1,n,1).shape)
        style_memory = self.film_style(
            self.memory_style.repeat(bs,1,1), 
            style_vec.repeat(1,64,1), 
        )
        motion_memory = torch.cat([base_memory, style_memory], dim=-1)  # [B,64,d]
        motion_memory = self.feature_fusion(motion_memory)
        
        audio2face_fea_proj = self.cross_attn(audio2face_fea_proj, motion_memory)
        audio2face_fea_proj = self.norm(audio2face_fea_proj)
        audio2face_fea_proj = self.position_embeddings(audio2face_fea_proj)
        
        decode_face = x
        # decode_face = self.face_motion_cross_audio(x, audio2face_fea_proj, emb)
        for decoder_layer in self.face_motion_cross_audio:
            decode_face = decoder_layer(
                decode_face,
                audio2face_fea_proj,
                emb,
                cross_cond_2=masked_motion,
            )
        face_latent = self.face_out_proj(decode_face)
        return face_latent

    def inference(self, audio, speaker_id, vq_model=None, masked_motion=None, mask=None, noise_scheduler=None):
        self.inference_pipeline.setup_scheduler(noise_scheduler)
        # generate default mask and masked motion if not provided
        # length = audio.shape[1] * 30 // 16000
        length = masked_motion.shape[1]
        bs = audio.shape[0]
        # print(length, masked_motion.shape)
        fake_motion = torch.zeros(bs, length, self.cfg.vae_codebook_size).to(audio.device)
        if masked_motion is not None:
            fake_motion[:, :masked_motion.shape[1]] = masked_motion 
        masked_motion = fake_motion
        
        generator = torch.Generator(device=audio.device)
        generator.manual_seed(self.cfg.seed)

        fake_mask = torch.ones_like(masked_motion)
        if mask is not None:
            fake_mask[:, :mask.shape[1]] = mask 
        mask = fake_mask
        # Autoregressive inference
        bs, total_len, c = masked_motion.shape
        window = self.cfg.pose_length
        pre_frames = self.cfg.seed_frames
        rounds = (total_len - pre_frames) // (window - pre_frames)
        remain = (total_len - pre_frames) % (window - pre_frames)
        
        rec_all_face = []
        last_motion = masked_motion[:, :pre_frames, :]
        for i in range(rounds):
            start_idx = i*(window - pre_frames)
            end_idx = start_idx + window

            window_mask = mask[:, start_idx:end_idx, :].clone()
            window_motion = masked_motion[:, start_idx:end_idx, :].clone()
            window_motion[:, :pre_frames, :] = last_motion
            # window_motion[:, :pre_frames, :] = torch.where(
            #     (window_mask[:, :pre_frames, :] == 0),
            #     masked_motion[:, start_idx:start_idx+pre_frames, :],
            #     last_motion,
            # )
            window_mask[:, :pre_frames, :] = 0

            audio_slice_len = (end_idx - start_idx)*(16000//30)
            audio_slice = audio[:, start_idx*(16000//30) : start_idx*(16000//30)+audio_slice_len]
            # print(i, audio_slice.shape, speaker_id.shape, window_motion.shape, window_mask.shape)
            
            bs, t, _ = window_mask.shape
            x_init = torch.randn((bs, t, self.cfg.vae_codebook_size), dtype=torch.float32, device=window_mask.device)
            # print(self.cfg.ode_step_size)
            face_latent = self.inference_pipeline(
                num_inference_steps=self.cfg.denoising_steps,
                scheduler=noise_scheduler,
                device=audio.device,
                generator=generator,
                audio=audio_slice, speaker_id=speaker_id, masked_motion=window_motion, mask=window_mask, use_audio=True)
            
            if i == 0:
                rec_all_face.append(face_latent[:, :-pre_frames, :])
            else:
                # blending 
                blend_factor = 1 / (pre_frames + 2)
                face_latent_to_blend = face_latent[:, :pre_frames, :]
                last_motion_to_blend = window_motion[:, :pre_frames, :]
                for i in range(pre_frames):
                    blend_ratio = blend_factor * (i + 1)
                    face_latent_to_blend[:, i, :] = (1 - blend_ratio) * face_latent_to_blend[:, i, :] + blend_ratio * last_motion_to_blend[:, i, :]
                face_latent[:, :pre_frames, :] = face_latent_to_blend
                rec_all_face.append(face_latent[:, :-pre_frames, :])
            last_motion = face_latent[:, -pre_frames:, :]
            # print(face_latent[:, :-pre_frames, :].shape)

        if remain > pre_frames:
            final_start = rounds*(window - pre_frames)
            final_end = final_start + pre_frames + remain

            final_mask = mask[:, final_start:final_end, :].clone()
            final_motion = masked_motion[:, final_start:final_end, :].clone()
            final_motion[:, :pre_frames, :] = last_motion
            # torch.where(
            #     (final_mask[:, :pre_frames, :] == 0),
            #     masked_motion[:, final_start:final_start+pre_frames, :],
            #     last_motion,
            # )
            final_mask[:, :pre_frames, :] = 0

            audio_slice_len = (final_end - final_start)*(16000//30)
            audio_slice = audio[:, final_start*(16000//30) : final_start*(16000//30)+audio_slice_len]
            bs, t, _ = final_mask.shape
            x_init = torch.randn((bs, t, self.cfg.vae_codebook_size), dtype=torch.float32, device=window_mask.device)
            
            face_latent = self.inference_pipeline(
                num_inference_steps=self.cfg.denoising_steps,
                scheduler=noise_scheduler,
                device=audio.device,
                generator=generator,
                audio=audio_slice, speaker_id=speaker_id, masked_motion=final_motion, mask=final_mask, use_audio=True)
            
            blend_factor = 1 / (pre_frames + 2)
            face_latent_to_blend = face_latent[:, :pre_frames, :]
            last_motion_to_blend = final_motion[:, :pre_frames, :]
            for i in range(pre_frames):
                blend_ratio = blend_factor * (i + 1)
                face_latent_to_blend[:, i, :] = (1 - blend_ratio) * face_latent_to_blend[:, i, :] + blend_ratio * last_motion_to_blend[:, i, :]
            face_latent[:, :pre_frames, :] = face_latent_to_blend
            rec_all_face.append(face_latent)
        rec_all_face = torch.cat(rec_all_face, dim=1) 
        # print(rec_all_face.shape)
        return rec_all_face