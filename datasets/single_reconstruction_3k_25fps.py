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

args = argparse.ArgumentParser()
args.add_argument("--cache_dir", type=str, default="/home/weili/haiyang/PantoMatrix/HDTF/cache_latent_v4/")
args.add_argument("--save_dir", type=str, default="/home/weili/haiyang/PantoMatrix/HDTF/test_reconstructions_v4/")
args = args.parse_args()

transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.5], [0.5])
])

motion_latent_dir = args.cache_dir
gt_latent_dir = "/home/weili/haiyang/PantoMatrix/HDTF/cache_latent_v4/"
video_dir = "/home/weili/haiyang/PantoMatrix/HDTF/cache_merge_v6/"
save_path = args.save_dir
os.makedirs(save_path, exist_ok=True)

config = OmegaConf.load("/home/weili/haiyang/PantoMatrix/datasets/audio_head_animator_v4.yaml")
module = instantiate(config.model, instantiate_module=False)
model = module(config=config)
checkpoint = torch.load(config.resume_ckpt)
model.load_state_dict(checkpoint["state_dict"], strict=False)
model.eval().to("cuda")
motion_encoder = model.motion_encoder
flow_estimator = model.flow_estimator
face_generator = model.face_generator
face_encoder = model.face_encoder

for latent_file in tqdm(os.listdir(motion_latent_dir)):
    if not latent_file.endswith(".npy"):
        continue
    file_name = latent_file[:-15]
    gt_latent = np.load(os.path.join(gt_latent_dir, file_name + ".npz"), allow_pickle=True)["random_data"]
    gt_latent = torch.from_numpy(gt_latent).to("cuda")
    tgt_latent = np.load(os.path.join(motion_latent_dir, latent_file))
    tgt_latent = torch.from_numpy(tgt_latent).to("cuda").squeeze(0).float()
    aligned_video = os.path.join(video_dir, file_name + ".mp4")
    load_video = VideoReader(aligned_video)
    source_img = load_video[0].asnumpy()
    source_img = transform(Image.fromarray(source_img)).unsqueeze(0).to("cuda")
    src_latent = gt_latent[0:1]
    with torch.no_grad():
        face_feat = face_encoder(source_img)
        all_recon_imgs = []
        all_recon_imgs.append(source_img)
        for i in tqdm(range(1, tgt_latent.shape[0])):
            tgt_latent_refine = flow_estimator(src_latent, tgt_latent[i:i+1])
            recon_imgs = face_generator(tgt_latent_refine, face_feat)
            all_recon_imgs.append(recon_imgs)
    recon_imgs = torch.cat(all_recon_imgs, dim=0)
    video_pred = recon_imgs.permute(0, 2, 3, 1).cpu().numpy()
    video_pred = np.clip((video_pred + 1) / 2 * 255, 0, 255).astype("uint8")
    save_video_path = os.path.join(save_path, f"{file_name}_recon.mp4")
    with imageio.get_writer(save_video_path, fps=25) as writer:
        for i in range(len(video_pred)):
            writer.append_data(video_pred[i])
    audio_path = os.path.join(gt_latent_dir.replace("cache_latent", "cache_audio"), file_name + ".wav")
    if os.path.exists(audio_path):
        video_clip = mp.VideoFileClip(save_video_path)
        audio_clip = mp.AudioFileClip(audio_path)
        video_with_audio = video_clip.set_audio(audio_clip)
        final_output_path = os.path.join(save_path, f"{file_name}.mp4")
        video_with_audio.write_videofile(final_output_path, codec="libx264", audio_codec="aac")
        video_clip.close()
        audio_clip.close()
os.system(f"rm -rf {save_path}/*_recon.mp4")