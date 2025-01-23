import os
import shutil
import argparse
import random
import numpy as np
from datetime import datetime
from tqdm import tqdm
import inspect
import importlib
import copy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
from diffusers.optimization import get_scheduler
from diffusers import DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers import DiffusionPipeline
from diffusers.schedulers import (
    PNDMScheduler,
    LMSDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
from diffusers.utils import BaseOutput

from omegaconf import OmegaConf
from utils.tools import compute_snr
from utils.draw_pose import draw_single_video, merge_single_videos_in_one_row, merge_single_videos_in_one_column, draw_overlay

from dataclasses import dataclass
from typing import Callable, List, Optional, Union

def train_val_fn(cfg, batch, model, device, noise_scheduler, mode="train", optimizer=None, lr_scheduler=None, max_grad_norm=1.0, **kwargs):
    if mode == "train":
        model.train()
        torch.set_grad_enabled(True)
        optimizer.zero_grad()
    else:
        model.eval()
        torch.set_grad_enabled(False)

    motion = batch["motion"].to(device)
    cond_motion = batch["cond_motion"].to(device)
    vaild_mask = batch["vaild_mask"].to(device)
    vaild_mask = vaild_mask.reshape(vaild_mask.shape[0],vaild_mask.shape[1],-1)
    vaild_mask[:, 122:124] = 0

    latents = motion
    noise = torch.randn_like(latents)
    if cfg.noise_offset > 0:
        noise += cfg.noise_offset * torch.randn(
                (latents.shape[0], latents.shape[1], 1),
                device=latents.device,
            )
    bsz = latents.shape[0]
    timesteps = torch.randint(
            0,
            noise_scheduler.num_train_timesteps,
            (bsz,),
            device=latents.device,
        )
    timesteps = timesteps.long()
    noisy_latents = noise_scheduler.add_noise(
            latents, noise, timesteps
    )
    model_pred = model(x=noisy_latents, timesteps=timesteps, y={"cond_motion": cond_motion})

    if noise_scheduler.prediction_type == "epsilon":
        target = noise
        # print("hrere")
    elif noise_scheduler.prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(
            latents, noise, timesteps
        )
    elif noise_scheduler.prediction_type == "sample":
        target = motion
    else:
        raise ValueError(
            f"Unknown prediction type {noise_scheduler.prediction_type}"
        )

    denoising_loss = denoising_loss_fn(cfg, model_pred*vaild_mask, target*vaild_mask, noise_scheduler, timesteps)

    loss_dict = {
        "denoising": denoising_loss,
    }
    loss = sum(loss_dict.values())
    loss_dict["loss"] = loss

    if mode == "train":
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        lr_scheduler.step()

    return loss_dict

@dataclass
class Pose2PosePipelineOutput(BaseOutput):
    poses: Union[torch.Tensor, np.ndarray]

class Pose2PosePipeline(DiffusionPipeline):
    _optional_components = []
    def __init__(
        self,
        model,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ):
        super().__init__()
        self.register_modules(
            model=model,
            scheduler=scheduler,
        )

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.model, "_hf_hook"):
            return self.device
        for module in self.model.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

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
        cond_motion,
        num_inference_steps,
        device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.0,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        dtype = cond_motion.dtype
        batch_size = cond_motion.shape[0]

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        latents = randn_tensor(
            cond_motion.shape, generator=generator, device=device, dtype=dtype
        )
        latents = latents * self.scheduler.init_noise_sigma

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Create a batch of timesteps
                t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

                latent_model_input = self.scheduler.scale_model_input(
                    latents, t
                )
                noise_pred = self.model(
                    x=latent_model_input,
                    timesteps=t_batch,
                    y={"cond_motion": cond_motion}
                )
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
        output = latents
        return Pose2PosePipelineOutput(poses=output)


def test_fn(cfg, model, device, test_dataset, test_loader, val_noise_scheduler, iteration, test_path, **kwargs):
    torch.set_grad_enabled(False)
    actual_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    actual_model.eval()
    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.seed)
    pipeline = Pose2PosePipeline(
        model=actual_model,
        scheduler=val_noise_scheduler,
    )
    pipeline = pipeline.to(device)
    
    all_l1_loss = []
    for idx, batch in enumerate(test_loader):
        if idx == 64: break
        motion = batch["motion"].to(device)
        cond_motion = batch["cond_motion"].to(device)
        vaild_mask = batch["vaild_mask"].to(device)
        invalid_mask = ~vaild_mask.squeeze(0)[:, :, :2]

        pose_tensor = pipeline(
            cond_motion,
            cfg.validation.denoising_steps,
            device,
            generator=generator,
        ).poses
        
        # evaluation, mse loss to gt
        pose_np = motion.cpu().squeeze(0).numpy()[:, :120]
        pose_np = pose_np.reshape(pose_np.shape[0], 60, 2)
        pose_np = pose_np * test_dataset.std[:60] + test_dataset.mean[:60]
        pose_np[:, 0:1, :] = pose_np[:, 0:1, :] + pose_np[:, 1:2, :]
        pose_np[:, 2:, :] = pose_np[:, 2:, :] + pose_np[:, 1:2, :]
        # print(pose_np.shape, invalid_mask.shape)
        pose_np[invalid_mask.cpu().numpy()] = -1 
        pose_np = np.concatenate([pose_np, np.zeros((pose_np.shape[0], 68, 2))], 1)
        np.save(test_path+f"gt_{idx}.npy", pose_np)

        linear_motion = np.linspace(pose_np[0], pose_np[-1], pose_np.shape[0])
        np.save(test_path+f"linear_{idx}.npy", linear_motion)

        pred_pose_np = pose_tensor.cpu().squeeze(0).numpy()[:, :120]
        pred_pose_np = pred_pose_np.reshape(pred_pose_np.shape[0], 60, 2)
        pred_pose_np = pred_pose_np * test_dataset.std[:60] + test_dataset.mean[:60]
        pred_pose_np[:, 0:1, :] = pred_pose_np[:, 0:1, :] + pred_pose_np[:, 1:2, :]
        pred_pose_np[:, 2:, :] = pred_pose_np[:, 2:, :] + pred_pose_np[:, 1:2, :]
        pred_pose_np[invalid_mask.cpu().numpy()] = -1
        pred_pose_np = np.concatenate([pred_pose_np, np.zeros((pred_pose_np.shape[0], 68, 2))], 1)
        np.save(test_path+f"pred_{idx}.npy", pred_pose_np)

        cond_pose_np = cond_motion.cpu().squeeze(0).numpy()[:, :120]
        cond_pose_np = cond_pose_np.reshape(cond_pose_np.shape[0], 60, 2)
        cond_pose_np = cond_pose_np * test_dataset.std[:60] + test_dataset.mean[:60]
        cond_pose_np[:, 0:1, :] = cond_pose_np[:, 0:1, :] + cond_pose_np[:, 1:2, :]
        cond_pose_np[:, 2:, :] = cond_pose_np[:, 2:, :] + cond_pose_np[:, 1:2, :]
        cond_pose_np[invalid_mask.cpu().numpy()] = -1
        cond_pose_np = np.concatenate([cond_pose_np, np.zeros((cond_pose_np.shape[0], 68, 2))], 1)
        np.save(test_path+f"cond_{idx}.npy", cond_pose_np)

        l1_loss = np.abs(pose_np - pred_pose_np).mean()
        all_l1_loss.append(l1_loss)
       
        # visualization
        draw_single_video(test_path+f"gt_{idx}.mp4", test_path+f"gt_{idx}.npy", draw_face=False)
        draw_single_video(test_path+f"pred_{idx}.mp4", test_path+f"pred_{idx}.npy", draw_face=False)
        draw_single_video(test_path+f"cond_{idx}.mp4", test_path+f"cond_{idx}.npy", draw_face=False)
        draw_single_video(test_path+f"linear_{idx}.mp4", test_path+f"linear_{idx}.npy", draw_face=False)

        draw_overlay(test_path+f"gt_{idx}.mp4", test_path+f"pred_{idx}.mp4", test_path+f"overlay_gt_pred_{idx}.mp4")
        draw_overlay(test_path+f"linear_{idx}.mp4", test_path+f"pred_{idx}.mp4", test_path+f"overlay_linear_pred_{idx}.mp4")
        # draw_overlay(test_path+f"gt_{idx}.mp4", test_path+f"cond_{idx}.mp4", test_path+f"overlay_gt_cond_{idx}.mp4")
        # draw_overlay(test_path+f"pred_{idx}.mp4", test_path+f"cond_{idx}.mp4", test_path+f"overlay_pred_cond_{idx}.mp4")
        merge_single_videos_in_one_row(test_path+f"merge_{idx}.mp4", [test_path+f"cond_{idx}.mp4", test_path+f"linear_{idx}.mp4", test_path+f"pred_{idx}.mp4", test_path+f"gt_{idx}.mp4", test_path+f"overlay_gt_pred_{idx}.mp4", test_path+f"overlay_linear_pred_{idx}.mp4"])
        # merge_single_videos_in_one_row(test_path+f"overlay_merge_{idx}.mp4", [test_path+f"overlay_gt_pred_{idx}.mp4", test_path+f"overlay_gt_cond_{idx}.mp4", test_path+f"overlay_pred_cond_{idx}.mp4"])
        # merge_single_videos_in_one_column(test_path+f"merge_all_{idx}.mp4", [test_path+f"merge_{idx}.mp4", test_path+f"overlay_merge_{idx}.mp4"])
    
    metrics = {"l1_loss": np.mean(all_l1_loss)}
    print(f"Test Metrics at Iteration {iteration}:")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")
    return metrics

def denoising_loss_fn(cfg, model_pred, target, noise_scheduler, timesteps):
    # print(model_pred.shape, target.shape)
    if cfg.snr_gamma == 0:
        loss = F.mse_loss(
            model_pred.float(), target.float(), reduction="mean"
        )
    else:
        snr = compute_snr(noise_scheduler, timesteps)
        if noise_scheduler.config.prediction_type == "v_prediction":
            # Velocity objective requires that we add one to SNR values before we divide by them.
            snr = snr + 1
        mse_loss_weights = (
            torch.stack(
                [snr, cfg.snr_gamma * torch.ones_like(timesteps)], dim=1
            ).min(dim=1)[0]
            / snr
        )
        # print(model_pred.shape, target.shape)
        loss = F.mse_loss(
            model_pred.float(), target.float(), reduction="none"
        )
        loss = (
            loss.mean(dim=list(range(1, len(loss.shape))))
            * mse_loss_weights
        )
        loss = loss.mean()
    return loss

def main(cfg):
    # environment init
    local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.distributed.init_process_group(backend="nccl")
    seed_everything(cfg.seed)
    experiment_ckpt_dir = experiment_log_dir = os.path.join(cfg.output_dir, cfg.exp_name)
    
    # model init
    model = init_class(cfg.model.name_pyfile, cfg.model.class_name, cfg).to(device)
    for param in model.parameters():
        param.requires_grad = True  
 
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
        # broadcast_buffers=False,
    )

    # optimizer init 
    if cfg.solver.use_8bit_adam:
        pass
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.solver.learning_rate,
        betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2),
        weight_decay=cfg.solver.adam_weight_decay,
        eps=cfg.solver.adam_epsilon,)
    lr_scheduler = get_scheduler(
        cfg.solver.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.solver.lr_warmup_steps
        * cfg.solver.gradient_accumulation_steps,
        num_training_steps=cfg.solver.max_train_steps
        * cfg.solver.gradient_accumulation_steps,
    )

    sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
    if cfg.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            # prediction_type="v_prediction",
        )
    val_noise_scheduler = DDIMScheduler(**sched_kwargs)
    sched_kwargs.update({"beta_schedule": "scaled_linear"})
    train_noise_scheduler = DDIMScheduler(**sched_kwargs)

    # dataset init
    train_dataset = init_class(cfg.data.name_pyfile, cfg.data.class_name, cfg, split='train')
    test_dataset = init_class(cfg.data.name_pyfile, cfg.data.class_name, cfg, split='test')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=cfg.data.train_bs, sampler=train_sampler, drop_last=True, num_workers=4)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=1, sampler=test_sampler, drop_last=False, num_workers=4)

    if local_rank == 0:
        run_time = datetime.now().strftime("%Y%m%d-%H%M")
        wandb.init(
            project=cfg.wandb_project,
            name=cfg.exp_name + "_" + run_time,
            entity=cfg.wandb_entity,
            dir=cfg.wandb_log_dir,
            config=OmegaConf.to_container(cfg)
        )
    
    num_epochs = cfg.solver.max_train_steps // len(train_loader) + 1
    iteration = 0
    val_best = {}
    test_best = {}
    
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            loss_dict = train_val_fn(
                cfg, batch, model, device, train_noise_scheduler, mode="train", optimizer=optimizer, lr_scheduler=lr_scheduler
            )
            if local_rank == 0 and iteration % cfg.log_period == 0:
                for key, value in loss_dict.items():
                    wandb.log({f"train/{key}": value}, step=iteration)
                loss_message = ", ".join([f"{k}: {v:.6f}" for k, v in loss_dict.items()])
                print(f"Epoch {epoch} [{i}/{len(train_loader)}] - {loss_message}")

            if local_rank == 0 and iteration % cfg.validation.val_loss_steps == 0:
                val_loss_dict = {}
                val_batches = 0
                for batch in tqdm(test_loader):
                    loss_dict = train_val_fn(
                        cfg, batch, model, device, val_noise_scheduler, mode="val"
                    )
                    for k, v in loss_dict.items():
                        if k not in val_loss_dict:
                            val_loss_dict[k] = 0
                        val_loss_dict[k] += v.item()
                    val_batches += 1
                    if val_batches == 10:
                        break
                val_loss_mean_dict = {k: v / val_batches for k, v in val_loss_dict.items()}
                for k, v in val_loss_mean_dict.items():
                    if k not in val_best or v < val_best[k]["value"]:
                        val_best[k] = {"value": v, "iteration": iteration}
                        if "denoising" in k:
                            checkpoint_path = os.path.join(experiment_ckpt_dir, f"ckpt_{k}")
                            os.makedirs(checkpoint_path, exist_ok=True)
                            torch.save({
                                'iteration': iteration,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                            }, os.path.join(checkpoint_path, "ckpt.pth"))

                    print(f"Val [{iteration}] - {k}: {v:.6f} (best: {val_best[k]['value']:.6f} at {val_best[k]['iteration']})")
                    wandb.log({f"val/{k}": v}, step=iteration)
        
                checkpoint_path = os.path.join(experiment_ckpt_dir, f"checkpoint_{iteration}")
                os.makedirs(checkpoint_path, exist_ok=True)
                torch.save({
                    'iteration': iteration,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                }, os.path.join(checkpoint_path, "ckpt.pth"))
                checkpoints = [d for d in os.listdir(experiment_ckpt_dir) if os.path.isdir(os.path.join(experiment_ckpt_dir, d)) and d.startswith("checkpoint_")]
                checkpoints.sort(key=lambda x: int(x.split("_")[1]))
                if len(checkpoints) > 3:
                    for ckpt_to_delete in checkpoints[:-3]:
                        shutil.rmtree(os.path.join(experiment_ckpt_dir, ckpt_to_delete))

            if local_rank == 0 and iteration % cfg.validation.validation_steps == 0:
                test_path = os.path.join(experiment_ckpt_dir, f"test_{iteration}") + "/"
                os.makedirs(test_path, exist_ok=True)
                test_metric_dict = test_fn(cfg, model, device, test_dataset, test_loader, val_noise_scheduler, iteration, test_path)
                for k, v in test_metric_dict.items():
                    if k not in test_best or v < test_best[k]["value"]:
                        test_best[k] = {"value": v, "iteration": iteration}  
                    print(f"Test [{iteration}] - {k}: {v:.6f} (best: {test_best[k]['value']:.6f} at {test_best[k]['iteration']})")
                    wandb.log({f"test/{k}": v}, step=iteration)
                video_for_log = []
                video_res_path = os.path.join(test_path)
                for mp4_file in os.listdir(video_res_path):
                    if mp4_file.endswith(".mp4"):
                        file_path = os.path.join(video_res_path, mp4_file)
                        log_video = wandb.Video(file_path, caption=f"{iteration:06d}-{mp4_file}", format="mp4")
                        video_for_log.append(log_video)
                wandb.log(
                  {"test/videos": video_for_log},
                  step=iteration
                )
            iteration += 1

    if local_rank == 0:
        wandb.finish()
    torch.distributed.destroy_process_group()

def init_class(module_name, class_name, config, **kwargs):
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)
    instance = model_class(config, **kwargs)
    return instance

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def prepare_all():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/train/stage2.yaml")
    parser.add_argument("--debug", action="store_true", help="Enable debugging mode")
    parser.add_argument('overrides', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config.endswith(".yaml"):
        config = OmegaConf.load(args.config)
        config.exp_name = args.config.split("/")[-1][:-5]
    else:
        raise ValueError("Unsupported config file format. Only .yaml files are allowed.")

    if args.debug:
        config.wandb_project = "debug"
        config.exp_name = "debug"
        config.solver.max_train_steps = 4

    if args.overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.overrides))
    
    os.environ["WANDB_API_KEY"] = config.wandb_key

    save_dir = os.path.join(config.output_dir, config.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'sanity_check'), exist_ok=True)

    config_path = os.path.join(save_dir, 'sanity_check', f'{config.exp_name}.yaml')
    with open(config_path, 'w') as f:
        OmegaConf.save(config, f)

    current_dir = os.getcwd()
    sanity_check_dir = os.path.join(save_dir, 'sanity_check')
    for root, dirs, files in os.walk(current_dir):
        for file in files:
            if file.endswith(".py"):
                full_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_file_path, current_dir)
                dest_path = os.path.join(sanity_check_dir, relative_path)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy(full_file_path, dest_path)
    return config

if __name__ == "__main__":
    config = prepare_all()
    main(config)