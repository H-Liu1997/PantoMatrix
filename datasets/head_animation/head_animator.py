import torch
from torch import nn
import sys
from pathlib import Path
from einops import rearrange
import torch.nn.functional as F

sys.path.append(str(Path(__file__).parent.parent.parent))
from model.lightning.base_modules import BaseModule
from utils import instantiate
from model.head_animation.LIA.loss import VGGLoss

class HeadAnimatorModule(BaseModule):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.using_hybrid_mask = config.model.get("using_hybrid_mask", True)
        print(f'Using Hybird Mask: {self.using_hybrid_mask}')
        if not self.using_hybrid_mask:
            self.face_encoder = nn.Identity()
        
        self.criterion_recon = nn.L1Loss()
        self.criterion_L2 = nn.MSELoss()

        self.l_w_recon = config.loss.l_w_recon
        self.l_w_vgg = config.loss.l_w_vgg
        self.l_w_face = config.loss.get("l_w_face", 0)
        self.l_w_gan = config.loss.get("l_w_gan", 0)
        self.l_w_face_l2 = config.loss.get("l_w_face_l2", 0)
       
        # support GAN training & normal training
        self.automatic_optimization = False
        
    def configure_model(self):
        config = self.config
        self.motion_encoder = instantiate(config.model.motion_encoder)
        self.flow_estimator = instantiate(config.model.flow_estimator)
        self.face_generator = instantiate(config.model.face_generator)
        self.face_encoder = instantiate(config.model.face_encoder)

        if config.loss.l_w_vgg > 0 or config.loss.l_w_face > 0:
            self.criterion_vgg = VGGLoss()
            
        if config.loss.l_w_gan > 0:
            self.discriminator = instantiate(config.model.discriminator)
        
        if self.config.model.pretrained_ckpt is not None:
            checkpoint = torch.load(self.config.model.pretrained_ckpt)["state_dict"]
            ckpt = {}
            for k, v in checkpoint.items():
                if 'motion_encoder' in k:
                    ckpt[k.replace('motion_encoder.', '')] = v
            self.motion_encoder.load_state_dict(ckpt, strict=True)

            ckpt = {}
            for k, v in checkpoint.items():
                if 'flow_estimator' in k:
                    ckpt[k.replace('flow_estimator.', '')] = v
            self.flow_estimator.load_state_dict(ckpt, strict=True)

            ckpt = {}
            for k, v in checkpoint.items():
                if 'face_generator' in k:
                    ckpt[k.replace('face_generator.', '')] = v
            self.face_generator.load_state_dict(ckpt, strict=True)

            ckpt = {}
            for k, v in checkpoint.items():
                if 'face_encoder' in k:
                    ckpt[k.replace('face_encoder.', '')] = v
            self.face_encoder.load_state_dict(ckpt, strict=True)

         
    def motion_encode(self, source_img):
        latent_code, pyramid_feat = self.motion_encoder(source_img)
        return latent_code, pyramid_feat

    def forward(self, source_img, target_img, masked_source_img, masked_target_img):
        if self.using_hybrid_mask:
            tgt_latent, _ = self.motion_encoder(masked_target_img) # project target image to reference latent space
            src_latent, _ = self.motion_encoder(masked_source_img) # project source image to reference latent space

            tgt_latent = self.flow_estimator(src_latent, tgt_latent) # navigate source to target in reference latent space

            face_feat = self.face_encoder(source_img) 
            recon_img = self.face_generator(tgt_latent, face_feat)
        else:
            tgt_latent, _ = self.motion_encoder(target_img) # project target image to reference latent space
            src_latent, face_feat = self.motion_encoder(source_img) # project source image to reference latent space

            tgt_latent = self.flow_estimator(src_latent, tgt_latent) # navigate source to target in reference latent space
            recon_img = self.face_generator(tgt_latent, face_feat)
            # import pdb; pdb.set_trace()

        return recon_img
    
    def compute_loss(self, img_target, img_target_recon, face_mask=None):
        
        l1_loss = self.l_w_recon * self.criterion_recon(img_target_recon, img_target)
        
        # Perceptual Loss
        if self.l_w_vgg > 0:
            # img_target_recon = F.interpolate(img_target_recon, size=(256, 256), mode='bilinear', align_corners=False)
            # img_target = F.interpolate(img_target, size=(256, 256), mode='bilinear', align_corners=False)
            vgg_loss, vgg_loss_dict = self.criterion_vgg(img_target_recon, img_target)
            vgg_loss = self.l_w_vgg * vgg_loss.mean()
        else:
            vgg_loss = torch.zeros(1).to(self.device)
        
        # Facial Experssion Perceptual Loss
        if face_mask is not None and self.l_w_face > 0:
            face_loss, face_vgg_loss_dict = self.criterion_vgg(img_target_recon, img_target, face_mask)
            face_loss = self.l_w_face * face_loss.mean()
        else:
            face_loss = torch.zeros(1).to(self.device)

        if face_mask is not None and self.l_w_face_l2 > 0:
            face_l2_loss = self.l_w_face_l2 * self.criterion_L2(img_target_recon*face_mask, img_target*face_mask)
        else:
            face_l2_loss = torch.zeros(1).to(self.device)

        loss = vgg_loss + l1_loss + face_loss + face_l2_loss
        loss_dict = {'loss': loss, 'l1_loss': l1_loss, 'face_l2_loss': face_l2_loss, 'vgg_loss': vgg_loss, 'face_loss': face_loss}
        return loss_dict
    
    def g_nonsaturating_loss(self, fake_pred):
        return F.softplus(-fake_pred).mean()
    
    def d_nonsaturating_loss(self, fake_pred, real_pred):
        real_loss = F.softplus(-real_pred)
        fake_loss = F.softplus(fake_pred)

        return real_loss.mean() + fake_loss.mean()
    
    def prepare_datapair(self, batch):
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

        ref_img_original = ref_img_original[:,None].repeat(1, target_vid_original.size(1), 1, 1, 1)
        ref_img_original = rearrange(ref_img_original, "b t c h w -> (b t) c h w")
        target_vid_original = rearrange(target_vid_original, "b t c h w -> (b t) c h w")

        return ref_img_original, target_vid_original, masked_ref_img, masked_target_vid
    
    def _step(self, batch):
        # get source-target image pair
        ref_img_original, target_vid_original, masked_ref_img, masked_target_vid = self.prepare_datapair(batch)
        
        # get reconstructed image
        predicted_img = self.forward(ref_img_original, target_vid_original, masked_ref_img, masked_target_vid)

        if self.l_w_face > 0 or self.l_w_face_l2 > 0:
            eye_mouth_mask_vid = batch['eye_mouth_mask_vid']
            eye_mouth_mask_past_frames = batch['eye_mouth_mask_past_frames']
            face_mask = torch.cat([eye_mouth_mask_vid, eye_mouth_mask_past_frames], dim=1)
            face_mask = rearrange(face_mask, "b t c h w -> (b t) c h w")

            loss_dict = self.compute_loss(target_vid_original, predicted_img, face_mask)

        else:
            loss_dict = self.compute_loss(target_vid_original, predicted_img)

        if self.l_w_gan > 0:
            optimizer_g, optimizer_d = self.optimizers()
        
            ## train generator
            # self.toggle_optimizer(optimizer_g)

            # adversarial loss
            pred_label = self.discriminator(predicted_img).reshape(-1)
            g_loss = self.l_w_gan * self.g_nonsaturating_loss(pred_label)

            loss_dict['loss'] += g_loss
            loss_dict['g_loss'] = g_loss
            
            optimizer_g.zero_grad()
            self.manual_backward(loss_dict['loss'])
            optimizer_g.step()
            # self.untoggle_optimizer(optimizer_g)

            # import pdb; pdb.set_trace()
            
            ## train discriminator
            # self.toggle_optimizer(optimizer_d)

            real_img_pred = self.discriminator(target_vid_original)
            recon_img_pred = self.discriminator(predicted_img.detach())

            d_loss = self.d_nonsaturating_loss(recon_img_pred, real_img_pred)
            
            optimizer_d.zero_grad()
            self.manual_backward(d_loss)
            optimizer_d.step()
            # self.untoggle_optimizer(optimizer_d)

            self.log("d_loss", d_loss, prog_bar=True)
        
        else:
            optimizer_g = self.optimizers()

            optimizer_g.zero_grad()
            self.manual_backward(loss_dict['loss'])
            optimizer_g.step()
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

    def training_step(self, batch, batch_idx):
        loss_dict = self._step(batch)
        return loss_dict['loss']

    def validation_step(self, batch, batch_idx):
        # get source-target image pair
        ref_img_original, target_vid_original, masked_ref_img, masked_target_vid = self.prepare_datapair(batch)
        
        # get reconstructed image
        predicted_img = self.forward(ref_img_original, target_vid_original, masked_ref_img, masked_target_vid)
        loss_dict = self.compute_loss(target_vid_original, predicted_img)

        self.log('val_recon_loss', loss_dict['l1_loss'], prog_bar=True)

        return loss_dict['l1_loss']

    def configure_optimizers(self):
        params_to_update = list(self.motion_encoder.parameters()) + list(self.flow_estimator.parameters()) + \
                           list(self.face_encoder.parameters()) + list(self.face_generator.parameters())
        params_to_update = [p for p in params_to_update if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params_to_update,
            lr=self.config.optimizer.lr,
            weight_decay=self.config.optimizer.weight_decay,
            betas=(self.config.optimizer.adam_beta1, self.config.optimizer.adam_beta2),
            eps=self.config.optimizer.adam_epsilon,
        )

        if self.l_w_gan > 0:
            optimizer_dis = torch.optim.AdamW(
                self.discriminator.parameters(),
                lr=self.config.optimizer.discriminator_lr,
                weight_decay=self.config.optimizer.weight_decay,
                betas=(self.config.optimizer.adam_beta1, self.config.optimizer.adam_beta2),
                eps=self.config.optimizer.adam_epsilon,
            )
            return [optimizer, optimizer_dis], []
        else:
            return [optimizer], []


if __name__ == "__main__":
    from model.head_animation.LIA.motion_encoder import MotionEncoder
    from model.head_animation.LIA.flow_estimator import FlowEstimator
    from model.head_animation.LIA.face_encoder import FaceEncoder
    from model.head_animation.LIA.face_generator import FaceGenerator
    from torchsummaryX import summary
    
    IMAGE_SIZE = 512
    latent_dim = 512

    encoder = MotionEncoder(latent_dim=latent_dim, size=IMAGE_SIZE)
    # summary(encoder, torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)) 
    
    motion_space=20
    flow_estimator = FlowEstimator(latent_dim=latent_dim, motion_space=motion_space) 
    # summary(flow_estimator, torch.zeros(1, latent_dim), torch.zeros(1, latent_dim)) 
    tgt_latent = flow_estimator(torch.zeros(1, latent_dim), torch.zeros(1, latent_dim))

    face_encoder = FaceEncoder(output_channels=latent_dim) 
    # summary(face_encoder, torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)) 
    feat = face_encoder(torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)) 
    # for fea in feat: print(fea.shape)

    face_generator = FaceGenerator(IMAGE_SIZE, latent_dim, channel_multiplier=1)
    face_generator(tgt_latent, feat)
    

    