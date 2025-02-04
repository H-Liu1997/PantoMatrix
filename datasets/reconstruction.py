import os
import numpy as np
import torch
from decord import VideoReader
from PIL import Image
import torchvision.transforms as T
import imageio
from omegaconf import OmegaConf
from utils import instantiate
from tqdm import tqdm
import moviepy.editor as mp
import argparse
import time

args = argparse.ArgumentParser()
args.add_argument("--cache_dir", type=str, default="/home/weili/haiyang/outputs/infp_audio_longer_20250124-0725/test_100000")
args.add_argument("--save_dir", type=str, default="/home/weili/haiyang/PantoMatrix/HDTF/test_reconstructions/")
args = args.parse_args()

transform = T.Compose([
    # T.Resize((512, 512), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize([0.5], [0.5])
])
    
motion_latent_dir = args.cache_dir
gt_latent_dir = "/home/weili/haiyang/PantoMatrix/HDTF/cache_latent/"
video_dir = "/mnt/weka/training_data_1/hdtf_full/videos_resampled/"
save_path = args.save_dir
os.makedirs(save_path, exist_ok=True)

config = OmegaConf.load("/home/weili/haiyang/PantoMatrix/datasets/audio_head_animator.yaml")
module = instantiate(config.model, instantiate_module=False)
model = module(config=config)
checkpoint = torch.load(config.resume_ckpt)
model.load_state_dict(checkpoint["state_dict"], strict=False)
model.eval().to("cuda")
print(f'Load weight from {config.resume_ckpt}')
motion_encoder = model.motion_encoder
flow_estimator = model.flow_estimator
face_generator = model.face_generator
face_encoder = model.face_encoder

for latent_file in tqdm(os.listdir(motion_latent_dir)):
    if not latent_file.endswith(".npy"):
        continue
    file_name = latent_file[:-15]
    
    # Time data loading
    t_start = time.time()
    gt_latent = np.load(os.path.join(gt_latent_dir, file_name + ".npz"), allow_pickle=True)["random_data"]
    gt_latent = torch.from_numpy(gt_latent).to("cuda")
    tgt_latent = np.load(os.path.join(motion_latent_dir, latent_file))
    tgt_latent = torch.from_numpy(tgt_latent).to("cuda").squeeze(0).float()
    
    aligned_video = os.path.join(video_dir, file_name + ".mp4")
    load_video = VideoReader(aligned_video)
    source_img = load_video[0].asnumpy()
    source_img = transform(Image.fromarray(source_img)).unsqueeze(0).to("cuda")
    source_video = load_video.get_batch(range(len(load_video))).asnumpy()
    source_video = torch.from_numpy(source_video).float().to("cuda")/255*2-1
    t_load = time.time() - t_start
    print(f"Data loading time: {t_load:.2f}s")
    
    src_latent = gt_latent[0:1]
    with torch.no_grad():
        # Time face encoding
        t_start = time.time()
        face_feat = face_encoder(source_img)
        t_face_encode = time.time() - t_start
        print(f"Face encoding time: {t_face_encode:.2f}s")
        
        all_recon_imgs = []
        all_gt_recon_imgs = []
        t_flow_total = 0
        t_gen_total = 0
        
        for i in tqdm(range(1, tgt_latent.shape[0])):
            # Time flow estimation
            t_start = time.time()
            tgt_latent_refine = flow_estimator(src_latent, tgt_latent[i:i+1])
            t_flow = time.time() - t_start
            t_flow_total += t_flow
            
            # Time face generation
            t_start = time.time()
            recon_imgs = face_generator(tgt_latent_refine, face_feat)
            t_gen = time.time() - t_start
            t_gen_total += t_gen
            
            all_recon_imgs.append(recon_imgs)
        
            # Ground truth reconstruction timing
            t_start = time.time()
            gt_latent_refine = flow_estimator(src_latent, gt_latent[i:i+1])
            recon_imgs = face_generator(gt_latent_refine, face_feat)
            all_gt_recon_imgs.append(recon_imgs)
            
        n_frames = tgt_latent.shape[0] - 1
        print(f"Average flow estimation time per frame: {t_flow_total/n_frames:.3f}s")
        print(f"Average face generation time per frame: {t_gen_total/n_frames:.3f}s")
    
    # Time video saving
    t_start = time.time()
    recon_imgs = torch.cat(all_recon_imgs, dim=0)
    gt_recon_imgs = torch.cat(all_gt_recon_imgs, dim=0)
    # Visualize and save ground truth, reference image, audio predictions, and reconstructed images in a 2x2 grid
    video_pred = recon_imgs.permute(0, 2, 3, 1).cpu().numpy()
    video_gt = gt_recon_imgs.permute(0, 2, 3, 1).cpu().numpy()
    ref_img_original = source_img[0].permute(1, 2, 0).cpu().numpy()
    source_video = source_video.permute(0, 2, 3, 1).cpu().numpy()

    # Normalize images to [0, 1] and scale to [0, 255] for saving
    video_pred = np.clip((video_pred + 1) / 2 * 255, 0, 255).astype("uint8")
    video_gt = np.clip((video_gt + 1) / 2 * 255, 0, 255).astype("uint8")
    ref_img_original = np.clip((ref_img_original + 1) / 2 * 255, 0, 255).astype("uint8")
    source_video = np.clip((source_video + 1) / 2 * 255, 0, 255).astype("uint8")

    save_video_path = os.path.join(save_path, f"{file_name}_recon.mp4")
    with imageio.get_writer(save_video_path, fps=30) as writer:
        for i in range(len(video_pred)):
            upper_row = np.concatenate([source_video[i], video_gt[i]], axis=1)
            lower_row = np.concatenate([ref_img_original, video_pred[i]], axis=1)
            combined = np.concatenate([upper_row, lower_row], axis=0)
            writer.append_data(combined)
    t_save = time.time() - t_start
    print(f"Video saving time: {t_save:.2f}s")
    
    audio_path = os.path.join(gt_latent_dir.replace("cache_latent", "cache_audio"), file_name + ".wav")
    if os.path.exists(audio_path):
        t_start = time.time()
        video_clip = mp.VideoFileClip(save_video_path)
        audio_clip = mp.AudioFileClip(audio_path)
        video_with_audio = video_clip.set_audio(audio_clip)
        final_output_path = os.path.join(save_path, f"{file_name}_final.mp4")
        video_with_audio.write_videofile(final_output_path, codec="libx264", audio_codec="aac")
        video_clip.close()
        audio_clip.close()
        t_audio = time.time() - t_start
        print(f"Audio processing time: {t_audio:.2f}s")
    break
