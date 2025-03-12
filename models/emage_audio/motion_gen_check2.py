import math
import inspect
from omegaconf import OmegaConf
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel, Wav2Vec2Processor, Wav2Vec2Model, PretrainedConfig
from diffusers import DiffusionPipeline
from diffusers.utils.torch_utils import randn_tensor

from .motion_gen_utils import PeriodicPositionalEncoding, TimestepEncoding, FiLM, CrossAttention, SelfAttention, FeedforwardBlock, FilmTransformerDecoderLayer


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
        else:
            self.register_modules(scheduler=None)
    
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
                t_batch = torch.full((bs,), t, device=device, dtype=torch.long)
                latent_model_input = self.scheduler.scale_model_input(latents, t)
                noise_pred = self.model(
                    x=latent_model_input, 
                    t=t_batch, 
                    audio=model_extras["audio"], 
                    masked_motion=model_extras["masked_motion"], 
                    mask=model_extras["mask"], 
                    style_motion=model_extras["style_motion"],
                    style_motion_other=model_extras["style_motion_other"], 
                    audio_other=model_extras["audio_other"],
                    drop_other=model_extras["drop_other"]
                )
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)
        face_latent = latents
        return face_latent

class MotionGenModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # audio encoder
        self.audio_encoder_face = WrapedWav2Vec(layers=self.cfg.wav2vec_layer) # use how many transformer layers in wav2vec2      
        self.audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        
        # motion memory bank    
        self.memory_bank = nn.Parameter(torch.zeros(1, self.cfg.motion_bank_size, self.cfg.hidden_size))
        nn.init.normal_(self.memory_bank, 0, self.cfg.hidden_size**-0.5)
        self.memory_bank_other = nn.Parameter(torch.zeros(1, self.cfg.motion_bank_size, self.cfg.hidden_size))
        nn.init.normal_(self.memory_bank_other, 0, 3*self.cfg.hidden_size**-0.5)
        
        self.position_embeddings = PeriodicPositionalEncoding(
            self.cfg.hidden_size, period=self.cfg.pose_length, max_seq_len=self.cfg.pose_length
        )

        # motion decoder
        self.latent_proj_in = nn.Linear(self.cfg.vae_codebook_size, self.cfg.hidden_size)
        self.audio_proj_in = nn.Linear(768, self.cfg.hidden_size)
        self.style_proj_in = nn.Linear(self.cfg.vae_codebook_size, self.cfg.hidden_size)
        self.prev_motion_proj_in = nn.Linear(self.cfg.vae_codebook_size, self.cfg.hidden_size)
        self.style_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.cfg.hidden_size, nhead=self.cfg.heads, dim_feedforward=self.cfg.hidden_size*2
        )
        self.style_encoder = nn.TransformerEncoder(self.style_encoder_layer, num_layers=4)
        self.film_style = FiLM(self.cfg.hidden_size)
        self.film_style_other = FiLM(self.cfg.hidden_size)

        self.face_motion_cross_audio = nn.ModuleList(
            [
                FilmTransformerDecoderLayer(
                    self.cfg.hidden_size, self.cfg.hidden_size, self.cfg.heads, self.cfg.hidden_size*2, 0.1
                )
                for _ in range(self.cfg.layers)
            ]
        ) 
        self.latent_proj_out = nn.Linear(self.cfg.hidden_size, self.cfg.vae_codebook_size)
        self.time_embed = TimestepEncoding(self.cfg.hidden_size)
        
        self.audio_crosatt_memeory = CrossAttention(self.cfg.hidden_size, self.cfg.hidden_size, self.cfg.heads, 0.1)
        self.audio_crosatt_memeory_other = CrossAttention(self.cfg.hidden_size, self.cfg.hidden_size, self.cfg.heads, 0.1)
        self.norm1 = nn.LayerNorm(self.cfg.hidden_size)
        self.ffn = FeedforwardBlock(self.cfg.hidden_size, self.cfg.hidden_size*2)
        self.norm2 = nn.LayerNorm(self.cfg.hidden_size)
        self.inference_pipeline = Pose2PosePipeline(model=self)

    def forward(self, x, t, audio=None, masked_motion=None, mask=None, style_motion=None, audio_other=None, style_motion_other=None, drop_other=False, drop_style=False, drop_self=False, drop_style_other=False):
        invis_mask = mask
        cond_motion = masked_motion
        # latent
        x = self.latent_proj_in(x)
        bs, n, d = x.shape
        x = self.position_embeddings(x)
                
        # ------- audio feature ------- # 
        # print("audio shape: ", audio.shape)
        if audio is not None and not drop_self:
            audio_list = [i.cpu().numpy() for i in audio]
            inputs = self.audio_processor(audio_list, sampling_rate=16000, return_tensors="pt", padding=True).to(audio.device)
            audio2face_fea = self.audio_encoder_face(inputs.input_values)["high_level"]
            audio2face_fea = F.interpolate(
                audio2face_fea.transpose(1, 2), scale_factor=(self.cfg.pose_fps*64+33)/(50*64), mode="linear", align_corners=True
            ).transpose(1, 2)
            # print(audio2face_fea.shape, audio.shape)
            # assume normalized audio2face_fea
            audio2face_fea = audio2face_fea[:, :n]
            audio2face_fea_proj = self.audio_proj_in(audio2face_fea)
            audio2face_fea_proj = self.position_embeddings(audio2face_fea_proj)
        else:
            audio2face_fea_proj = None
        
        if audio_other is not None and not drop_other:
            audio_list_other = [i.cpu().numpy() for i in audio_other]
            inputs_other = self.audio_processor(audio_list_other, sampling_rate=16000, return_tensors="pt", padding=True).to(audio.device)
            audio2face_fea_other = self.audio_encoder_face(inputs_other.input_values)["high_level"]
            audio2face_fea_other = F.interpolate(
                audio2face_fea_other.transpose(1, 2), scale_factor=(self.cfg.pose_fps*64+33)/(50*64), mode="linear", align_corners=True
            ).transpose(1, 2)
            audio2face_fea_other = audio2face_fea_other[:, :n]
            audio2face_fea_proj_other = self.audio_proj_in(audio2face_fea_other)
            audio2face_fea_proj_other = self.position_embeddings(audio2face_fea_proj_other)
        else:
            audio2face_fea_proj_other = None
        
        # ---------- reference style motion  ---------- #
        drop_style_other = drop_style 
        if style_motion is not None and not drop_style:
            style_motion = style_motion[:, :n] 
            style_motion = self.style_proj_in(style_motion)
            style_motion = self.position_embeddings(style_motion)
            style_motion = self.style_encoder(style_motion)
            style_vec = torch.mean(style_motion, dim=1).unsqueeze(1)
        
        if style_motion_other is not None and not drop_style_other:
            style_motion_other = style_motion_other[:, :n] 
            style_motion_other = self.style_proj_in(style_motion_other)
            style_motion_other = self.position_embeddings(style_motion_other)
            style_motion_other = self.style_encoder(style_motion_other)
            style_vec_other = torch.mean(style_motion_other, dim=1).unsqueeze(1)

        # time embedding 
        if t.dim() == 0: t = t.unsqueeze(0)  # bs, 1    
        emb = self.time_embed(t).unsqueeze(1).repeat(1, n, 1)  # bs, 1, d -> bs, n, d
        
        # previous motion
        cond_motion = torch.where(invis_mask==1, 0.0, cond_motion)  # first 4 is gt
        cond_motion = cond_motion[:, :n]
        cond_motion = self.prev_motion_proj_in(cond_motion[:, :self.cfg.seed_frames, :])
        cond_motion = self.position_embeddings(cond_motion)
        
        # audio as query, motion memory as key and value
        memory_bank = self.memory_bank - torch.mean(self.memory_bank, dim=1, keepdim=True)
        motion_memory = memory_bank.repeat(bs, 1, 1)
        if style_motion is not None and not drop_style: motion_memory = self.film_style(motion_memory, style_vec) # x + (x-mu) / sigma
        if audio is not None and not drop_self: 
            audio2face_fea_proj = self.audio_crosatt_memeory(audio2face_fea_proj, motion_memory)
            audio_is_all_negative = torch.all(audio == -1, dim=1) 
            audio_is_all_negative = torch.where(audio_is_all_negative, 0, 1).unsqueeze(-1).unsqueeze(-1)
            audio2face_fea_proj = audio2face_fea_proj * audio_is_all_negative
        else: 
            audio2face_fea_proj = torch.zeros(bs, n, d).to(audio.device)
        
        memory_bank_other = self.memory_bank_other - torch.mean(self.memory_bank_other, dim=1, keepdim=True)
        motion_memory_other = memory_bank_other.repeat(bs, 1, 1)
        if style_motion_other is not None and not drop_style_other: motion_memory_other = self.film_style_other(motion_memory_other, style_vec_other)
        # max_diff = (motion_memory_other_after - motion_memory_other).max().item()
        # print(f"math check: drop_style_other is {drop_style_other}, if Ture, memory diff should be 0, the real value is {max_diff}")
        if audio_other is not None and not drop_other: 
            audio2face_fea_proj_other = self.audio_crosatt_memeory_other(audio2face_fea_proj_other, motion_memory_other)
            audio_other_is_all_negative = torch.all(audio_other == -1, dim=1)
            audio_other_is_all_negative = torch.where(audio_other_is_all_negative, 0, 1).unsqueeze(-1).unsqueeze(-1)
            # print("audio_other_is_zero_or_not: ", audio_other_is_all_negative.reshape(-1), audio2face_fea_proj_other.shape, audio_other_is_all_negative.shape)
            audio2face_fea_proj_other = audio2face_fea_proj_other * audio_other_is_all_negative
        else:
            audio2face_fea_proj_other = torch.zeros(bs, n, d).to(audio.device)
        # max_audio2face_fea_proj_other = audio2face_fea_proj_other.max().item()
        # print(f"math check: drop_other is {drop_other}, if Ture, audio2face_fea_proj_other should be 0, the real value is {max_audio2face_fea_proj_other}")
        # print("audio2face_fea_proj_other: ", audio2face_fea_proj_other.shape, audio2face_fea_proj.shape)
        min_len = min(audio2face_fea_proj.shape[1], audio2face_fea_proj_other.shape[1])
        audio2face_fea_proj = audio2face_fea_proj[:, :min_len]
        audio2face_fea_proj_other = audio2face_fea_proj_other[:, :min_len]
        audio2face_fea_proj = self.norm1(audio2face_fea_proj_other + audio2face_fea_proj)
        audio2face_fea_proj = self.ffn(audio2face_fea_proj)
        audio2face_fea_proj = self.norm2(audio2face_fea_proj)
        audio2face_fea_proj = self.position_embeddings(audio2face_fea_proj)
        
        decode_face = x
        for decoder_layer in self.face_motion_cross_audio:
            decode_face = decoder_layer(
                decode_face,
                audio2face_fea_proj,
                emb,
                cross_cond_2=cond_motion,
            )
        face_latent = self.latent_proj_out(decode_face)
        return face_latent

    def inference(self, audio, masked_motion=None, mask=None, noise_scheduler=None, style_motion=None, audio_other=None, style_motion_other=None):
        """
        Perform pseudo autoregressive inference for motion generation.

        Inference:
        - Audio length <= window size (e.g., 64 frames): infer in one step.
        - Audio length > window size: use sliding window with stride = window size. automatically blend the seed frames, use the last pre_frames as seed frames for the next window.

        Window size is from config, same as training.

        Args:
            audio (torch.Tensor): Shape (batch_size, n_frames). n_frames = past_audio_len + new_audio_len.
            audio_other (torch.Tensor): Same as audio.
            cond_motion (torch.Tensor): Shape (batch_size, t, d). t = past_motion_len + new_motion_len. New motion part is zeros.
            invis_mask (torch.Tensor): Shape (batch_size, t, d). 1 for invisible, 0 for visible. New motion part is zeros.
            style_motion (torch.Tensor): Shape (batch_size, t, d). Style reference, e.g., head motion.
            style_motion_other (torch.Tensor): Same as style_motion.

        Returns:
            torch.Tensor: Shape (batch_size, t, d). t = rec_past_motion and gen_new_motion.
        """
        if self.inference_pipeline.scheduler is None:
            raise ValueError("Inference pipeline is not set up the scheduler.")
        
        invis_mask = mask
        cond_motion = masked_motion
        
        drop_other = True if self.cfg.drop_other == 1 else False
        # print("drop_other: ", drop_other)
        length = cond_motion.shape[1]
        bs = audio.shape[0]
        fake_motion = torch.zeros(bs, length, self.cfg.vae_codebook_size).to(audio.device)
        if cond_motion is not None:
            fake_motion[:, :cond_motion.shape[1]] = cond_motion 
        cond_motion = fake_motion

        generator = torch.Generator(device=audio.device)
        generator.manual_seed(self.cfg.seed)

        fake_mask = torch.ones_like(cond_motion)
        if invis_mask is not None:
            fake_mask[:, :invis_mask.shape[1]] = invis_mask 
        invis_mask = fake_mask

        bs, total_len, c = cond_motion.shape
        window = self.cfg.pose_length
        pre_frames = self.cfg.seed_frames
        stride = window - pre_frames

        rec_all_face = []
        last_motion = cond_motion[:, :pre_frames, :]

        for i in range(0, total_len, stride):
            start_idx = i
            end_idx = min(start_idx + window, total_len)
            window_size = end_idx - start_idx

            window_mask = invis_mask[:, start_idx:end_idx, :].clone()
            window_motion = cond_motion[:, start_idx:end_idx, :].clone()
            # print("window_motion: ", window_motion.shape, last_motion.shape, pre_frames)
            window_motion[:, :pre_frames, :] = last_motion
            window_mask[:, :pre_frames, :] = 0  # Mask the seed frames

            window_style_motion = style_motion[:, start_idx:end_idx, :].clone() if style_motion is not None else None
            window_style_motion_other = style_motion_other[:, start_idx:end_idx, :].clone() if style_motion_other is not None else None

            audio_slice_len = window_size * (self.cfg.audio_fps // self.cfg.pose_fps)
            audio_slice_start = start_idx * (self.cfg.audio_fps // self.cfg.pose_fps)
            audio_slice = audio[:, audio_slice_start:audio_slice_start + audio_slice_len] if audio is not None else None
            audio_slice_other = audio_other[:, audio_slice_start:audio_slice_start + audio_slice_len] if audio_other is not None else None

            bs, t, _ = window_motion.shape
            if t <= 4: 
                rec_all_face.append(torch.zeros(bs, t, last_motion.shape[-1]).to(window_motion.device))
                break
            # this is single step inference    
            face_latent = self.inference_pipeline(
                num_inference_steps=self.cfg.denoising_steps,
                scheduler=noise_scheduler,
                device=audio.device,
                generator=generator,
                audio=audio_slice,
                masked_motion=window_motion,
                mask=window_mask,
                style_motion=window_style_motion,
                audio_other=audio_slice_other,
                style_motion_other=window_style_motion_other,
                drop_other=drop_other
            )
            if i == 0:
                rec_all_face.append(face_latent)
                last_motion = face_latent[:, -pre_frames:, :]
            else:
                if self.cfg.blend_frames:
                    blend_factor = 1 / (pre_frames + 2)
                    face_latent_to_blend = face_latent[:, :pre_frames, :]
                    last_motion_to_blend = window_motion[:, :pre_frames, :]
                    for j in range(pre_frames):
                        blend_ratio = blend_factor * (j + 1)
                        face_latent_to_blend[:, j, :] = (
                            (1 - blend_ratio) * face_latent_to_blend[:, j, :] +
                            blend_ratio * last_motion_to_blend[:, j, :]
                        )
                    face_latent[:, :pre_frames, :] = face_latent_to_blend
                new_frames = face_latent[:, pre_frames:, :]
                rec_all_face.append(new_frames)
                last_motion = face_latent[:, -pre_frames:, :]
            # print("face_latent: ", face_latent.shape, window_size, pre_frames)
            if face_latent.shape[1] < self.cfg.pose_length:
                break
        rec_all_face = torch.cat(rec_all_face, dim=1)
        return rec_all_face
    
    def one_clip_only_inference(self, 
        past_audio=None, new_audio=None, past_audio_other=None, new_audio_other=None, 
        past_motion=None, style_motion=None, style_motion_other=None, gen_frames=25,
    ):  
        bs, pt, d = past_motion.shape
        audio = torch.cat([past_audio, new_audio], dim=1) # 1, 5120 and 1, 16000
        if audio.shape[1] // int(self.cfg.audio_fps//sefl.cfg.pose_fps) < gen_frames:
            raise ValueError(f"Audio length {audio.shape[1]} is not enough for gen {gen_frames} frames.")
        if pt != self.cfg.seed_frames:
            raise ValueError(f"Past motion length {pt} is not equal to seed frames used in training {self.cfg.seed_frames}.")
        
        audio_other = torch.cat([past_audio_other, new_audio_other], dim=1) if past_audio_other is not None else None
        cond_motion = torch.cat([past_motion, torch.zeros(bs, gen_frames, d).to(past_motion.device)], dim=1)
        invis_mask = torch.cat([torch.zeros(bs, pt, d).to(past_motion.device), torch.ones(bs, gen_frames, d).to(past_motion.device)], dim=1)
        output = self.inference_pipeline(
                num_inference_steps=self.cfg.denoising_steps,
                device=audio.device,
                generator=generator,
                audio=audio,
                cond_motion=cond_motion,
                invis_mask=invis_mask,
                style_motion=style_motion,
                audio_other=audio_other,
                style_motion_other=style_motion_other
            )
        new_motion = output[:, pt:]
        new_past_motion = output[:, -pt:]
        new_past_audio = audio[:, -int(self.cfg.audio_fps//self.cfg.pose_fps*pt):] # hard coding here 640 = 16000/25
        if audio_other is not None:
            new_past_audio_other = audio_other[:, -int(self.cfg.audio_fps//self.cfg.pose_fps*pt):]
        return {
            "new_motion": new_motion, # bs, gen_frames, d
            "new_past_motion": new_past_motion,
            "new_past_audio": new_past_audio,
            "new_past_audio_other": new_past_audio_other
        } 