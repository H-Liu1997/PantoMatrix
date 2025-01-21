import torch
from torch import nn

from model.head_animation.head_animator import HeadAnimatorModule
from utils import instantiate

class AudioHeadAnimatorModule(HeadAnimatorModule):
    def __init__(self, config):
        super().__init__(config)
        self._get_scheduler()
    
    def configure_model(self):
        super().configure_model()
        self.motion_generator = instantiate(self.config.model.motion_generator)
        if self.config.model.motion_gen_ckpt is not None:
            checkpoint = torch.load(self.config.model.motion_gen_ckpt)["state_dict"]
            ckpt = {key.replace("model.", ""): value for key, value in checkpoint.items() if key.startswith("model.")}
            self.motion_generator.load_state_dict(ckpt, strict=False)
        self.motion_generator.to(dtype=eval(self.config.model.dtype))
    
    def _get_scheduler(self):
        if self.config.get("noise_scheduler", "flow") == "flow":
            from models.motion_scheduler import FlowMatchDiscreteScheduler
            self.train_noise_scheduler = FlowMatchDiscreteScheduler(
                shift=self.config.scheduler.flow_shift,
                reverse=self.config.scheduler.flow_reverse,
                solver=self.config.scheduler.flow_solver,
            )
            print('Using Flow scheduler now ! ')
        else:
            ddim_sched_kwargs = OmegaConf.to_container(self.config.ddim_scheduler_kwargs)
            self.train_noise_scheduler = DDIMScheduler(
                **ddim_sched_kwargs
            )
            print('Using DDIM scheduler now ! ')
            # raise ValueError(f"Invalid denoise type when training INFP model")

    @torch.no_grad()
    def motion_generate(self, masked_past_frames, audio_self, audio_other, video_length, guidance_scale=1.0):
        # 1. get data
        # masked_past_frames: [B, T, C, H, W]
        # past_latents: [B, T, C]
        past_latents = self.motion_encoder(masked_past_frames)
        B = audio_self.shape[0]
        C = past_latents.shape[-1]
        # 2. set timesteps
        self.train_noise_scheduler.set_timesteps(
            self.config.inference.num_inference_steps,
            device=self.device,
        )
        timesteps = self.train_noise_scheduler.timesteps
        num_warmup_steps = len(timesteps) - self.config.inference.num_inference_steps * self.train_noise_scheduler.order

        # 3.set inference latent
        generator = torch.manual_seed(torch.randint(0, 100000, (1,)).item())
        latents = (
            torch.randn((B, video_length, C), generator=generator, device=self.device, dtype=past_latents.dtype)
        )
        # Check existence to make it compatible with FlowMatchEulerDiscreteScheduler
        if hasattr(self.train_noise_scheduler, "init_noise_sigma"):
            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * self.train_noise_scheduler.init_noise_sigma
        do_classifier_free_guidance = False
        if guidance_scale > 1.0:
            do_classifier_free_guidance = True

        past_latents = (
            torch.cat([past_latents, past_latents])
            if do_classifier_free_guidance
            else past_latents
        )
        
        audio_self = (
            torch.cat([audio_self, audio_self])
            if do_classifier_free_guidance
            else audio_self
        )
        audio_other = (
            torch.cat([audio_other, audio_other])
            if do_classifier_free_guidance
            else audio_other
        )
        # 4. inference
        for i, t in enumerate(timesteps):
            print(i, t)
            if i < num_warmup_steps:
                continue

            # latent = self.train_noise_scheduler.scale_model_input(latent, t)
            
            motion_latents = latents.repeat(2 if do_classifier_free_guidance else 1, 1, 1, 1, 1)
            ts = torch.tensor([t] * motion_latents.shape[0], device=self.device, dtype=torch.float32)
            # forward
            with torch.autocast(
                    device_type="cuda", dtype=self.motion_generator.dtype, enabled=True
                ):
                noise_pred = self.motion_generator(
                    hidden_latents=motion_latents.to(dtype=torch.float32),
                    audio_self=audio_self.to(dtype=torch.float32),
                    audio_other=audio_other.to(dtype=torch.float32),
                    past_latents=past_latents.to(dtype=torch.float32),
                    num_frames=video_length,
                    n_past_frames=past_latents.shape[0] // B,
                    timestep=ts)
                print('noise_pred', noise_pred.shape)

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = self.train_noise_scheduler.step(
                noise_pred,
                t,
                latents,
            ).prev_sample  # outputs are prev_sample and pred_original_sample
            print('motion_latents', latents.shape)

        return latents

    def forward(self, source_img, masked_source_img, masked_past_frames, audio_self, audio_other, video_length, masked_target_img):
        if self.using_hybrid_mask:
            print("motion_generate with video length", video_length)
            tgt_motion_latent = self.motion_generate(masked_past_frames, audio_self, audio_other, video_length) # project target image to reference latent space
            tgt_motion_latent = tgt_motion_latent.view(-1, tgt_motion_latent.size(-1))
            print('tgt_motion_latent', tgt_motion_latent.shape)
            print("motion_encoder")
            src_latent = self.motion_encoder(masked_source_img) # project source image to reference latent space
            print('src_latent', src_latent.shape)

            tgt_latent = self.flow_estimator(src_latent, tgt_motion_latent) # navigate source to target in reference latent space
            face_feat = self.face_encoder(source_img) 
            recon_imgs = self.face_generator(tgt_latent, face_feat)
        else:
            tgt_latent = self.motion_generate(masked_past_frames, audio_self, audio_other, video_length) # project target image to reference latent space
            src_latent = self.motion_encoder(source_img) # project source image to reference latent space

            tgt_latent = self.flow_estimator(src_latent, tgt_latent) # navigate source to target in reference latent space
            face_feat = self.face_encoder(source_img) 
            recon_imgs = self.face_generator(tgt_latent, face_feat)

        return recon_imgs
    
    def _step(self, batch):

        optimizer_g, optimizer_d = self.optimizers()
        
        ## train generator
        self.toggle_optimizer(optimizer_g)
        masked_target_vid = batch['pixel_values_vid'] # this is a video batch: [B, T, C, H, W]
        masked_past_frames = batch['pixel_values_past_frames']
        masked_target_vid = torch.cat([masked_target_vid, masked_past_frames], dim=1)
        masked_ref_img = batch['pixel_values_ref_img']

        ref_img_original = batch['ref_img_original']
        target_vid_original = batch['pixel_values_vid_original']
        past_frames = batch['pixel_values_past_frames_original']
        target_vid_original = torch.cat([target_vid_original, past_frames], dim=1)
        
        # construct ref-tgt pairs
        masked_ref_img = masked_ref_img[:,None].repeat(1, masked_target_vid.size(1), 1, 1, 1)
        masked_ref_img = rearrange(masked_ref_img, "b t c h w -> (b t) c h w")
        masked_target_vid = rearrange(masked_target_vid, "b t c h w -> (b t) c h w")
        masked_past_frames = rearrange(masked_past_frames, "b t c h w -> (b t) c h w")

        ref_img_original = ref_img_original[:,None].repeat(1, target_vid_original.size(1), 1, 1, 1)
        ref_img_original = rearrange(ref_img_original, "b t c h w -> (b t) c h w")
        target_vid_original = rearrange(target_vid_original, "b t c h w -> (b t) c h w")
        
        audio_self = batch['target_wav_fea'] # torch.Size([1, 15, 130, 768])
        # TODO(wei): use audio other
        audio_other = torch.zeros_like(audio_self)
        # get reconstructed image
        predicted_img = self.forward(ref_img_original, masked_ref_img, masked_past_frames, audio_self, audio_other)

        if self.l_w_face > 0:
            eye_mouth_mask_vid = batch['eye_mouth_mask_vid']
            eye_mouth_mask_past_frames = batch['eye_mouth_mask_past_frames']
            face_mask = torch.cat([eye_mouth_mask_vid, eye_mouth_mask_past_frames], dim=1)
            face_mask = rearrange(face_mask, "b t c h w -> (b t) c h w")

            loss_dict = self.compute_loss(target_vid_original, predicted_img, face_mask)

        else:
            loss_dict = self.compute_loss(target_vid_original, predicted_img)

        if self.l_w_gan > 0:
            # adversarial loss
            pred_label = self.discriminator(predicted_img).reshape(-1)
            g_loss = self.l_w_gan * self.g_nonsaturating_loss(pred_label)

            loss_dict['loss'] += g_loss
            loss_dict['g_loss'] = g_loss

            self.manual_backward(loss_dict['loss'])
            optimizer_g.step()
            optimizer_g.zero_grad()
            self.untoggle_optimizer(optimizer_g)

            # import pdb; pdb.set_trace()
            
            ## train discriminator
            self.toggle_optimizer(optimizer_d)

            real_img_pred = self.discriminator(target_vid_original)
            recon_img_pred = self.discriminator(predicted_img.detach())

            d_loss = self.d_nonsaturating_loss(recon_img_pred, real_img_pred)

            self.manual_backward(d_loss)
            optimizer_d.step()
            optimizer_d.zero_grad()
            self.untoggle_optimizer(optimizer_d)

            self.log("d_loss", d_loss, prog_bar=True)
        
        else:
            self.manual_backward(loss_dict['loss'])
            optimizer_g.step()
            optimizer_g.zero_grad()
            self.untoggle_optimizer(optimizer_g)

        for k, v in loss_dict.items():
            self.log(k, v, prog_bar=True)

        
        if False:
            checkpoint = torch.load(self.config.model.pretrained_ckpt)["state_dict"]
            (self.motion_encoder.convs[0][0].weight - checkpoint['motion_encoder.convs.0.0.weight']).sum()
           
            # check vgg16 weight
            from torchvision import models
            vgg_model = models.vgg19(pretrained=True).cuda()
            vgg_params = []
            for p in vgg_model.parameters():
                vgg_params.append(p)

            (self.criterion_vgg.vgg.slice1[0].weight - vgg_params[0]).mean()
            (self.criterion_vgg.vgg.slice2[0].weight - vgg_params[2]).mean()
            import pdb; pdb.set_trace()
            

        return loss_dict