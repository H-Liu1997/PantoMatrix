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
args.add_argument("--cache_dir", type=str, default="/home/weili/haiyang/outputs/infp_audio_bs32_20250130-1314/test_0/wild")
args.add_argument("--save_dir", type=str, default="/home/weili/haiyang/outputs/infp_audio_bs32_20250130-1314/test_0/wild/cross_reconstruct")
args.add_argument("--fps", type=int, default=24)
args = args.parse_args()

transform = T.Compose([
    # T.Resize((512, 512), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize([0.5], [0.5])
])

def load_video_frames(video_path):
    vr = VideoReader(video_path)
    frames_np = vr.get_batch(range(len(vr))).asnumpy()
    frames_tensor = torch.from_numpy(frames_np).float().permute(0,3,1,2)/255*2-1
    return frames_tensor.to("cuda")

def generate_predictions(model, src_img, src_latent, tgt_latent):
    with torch.no_grad():
        feat = model.face_encoder(src_img)
        out = []
        for i in tqdm(range(tgt_latent.shape[0])):
            refine = model.flow_estimator(src_latent, tgt_latent[i:i+1])
            pred = model.face_generator(refine, feat)
            out.append(pred)
    return torch.cat(out, dim=0)

motion_latent_dir = args.cache_dir
gt_latent_dir = "/home/weili/haiyang/PantoMatrix/HDTF/cache_latent_v4/"
video_dir = "/home/weili/haiyang/PantoMatrix/HDTF/cache_ori_v4/"
save_path = args.save_dir
os.makedirs(save_path, exist_ok=True)

config = OmegaConf.load("/home/weili/haiyang/PantoMatrix/datasets/audio_head_animator.yaml")
module = instantiate(config.model, instantiate_module=False)
model = module(config=config)
ckpt = torch.load(config.resume_ckpt)
model.load_state_dict(ckpt["state_dict"], strict=False)
model.eval().to("cuda")

files = [f for f in os.listdir(motion_latent_dir) if f.endswith("_output.npz")]
for f in files:
    data = np.load(os.path.join(motion_latent_dir, f))
    gt_id = data["gt_video_id"].item()
    ref_id = data["ref_video_id"].item()
    no_latent = torch.from_numpy(data["no_style"]).float().to("cuda").squeeze(0)
    self_latent = torch.from_numpy(data["self_style"]).float().to("cuda").squeeze(0)
    cross_latent = torch.from_numpy(data["cross_style"]).float().to("cuda").squeeze(0)
    ref_latent = torch.from_numpy(data["ref_style"]).float().to("cuda").squeeze(0)
    print(no_latent.shape, self_latent.shape, cross_latent.shape, ref_latent.shape)
    vid1_gt_path = os.path.join(video_dir, gt_id + ".mp4")
    vid2_gt_path = os.path.join(video_dir, ref_id + ".mp4")
    if not (os.path.exists(vid1_gt_path) and os.path.exists(vid2_gt_path)):
        continue
    vid1_video = load_video_frames(vid1_gt_path)
    vid2_video = load_video_frames(vid2_gt_path)
    ref_img = vid1_video[0:1]
    gt_latent_1 = np.load(os.path.join(gt_latent_dir, gt_id + ".npz"), allow_pickle=True)["random_data"]
    gt_latent_2 = ref_latent # np.load(os.path.join(gt_latent_dir, ref_id + ".npz"), allow_pickle=True)["random_data"]
    src_latent_1 = torch.from_numpy(gt_latent_1).float().to("cuda")[0:1]
    # src_latent_2 = torch.from_numpy(gt_latent_2).float().to("cuda")[0:1]
    vid1_recon = generate_predictions(model, ref_img, src_latent_1, torch.from_numpy(gt_latent_1).float().to("cuda"))
    vid2_recon = generate_predictions(model, ref_img, src_latent_1, ref_latent)
    
    vid1_no = generate_predictions(model, ref_img, src_latent_1, no_latent)
    vid1_self = generate_predictions(model, ref_img, src_latent_1, self_latent)
    vid1_cross = generate_predictions(model, ref_img, src_latent_1, cross_latent)
    # vid1_ref = generate_predictions(model, ref_img, src_latent_1, ref_latent)
    frames_len = min(
        vid1_video.shape[0],
        vid2_video.shape[0],
        vid1_recon.shape[0],
        vid2_recon.shape[0],
        vid1_no.shape[0],
        vid1_self.shape[0],
        vid1_cross.shape[0],
    )
    vid1_gt_np = (vid1_video.permute(0,2,3,1).cpu().numpy()+1)/2*255
    vid2_gt_np = (vid2_video.permute(0,2,3,1).cpu().numpy()+1)/2*255
    vid1_rec_np = (vid1_recon.permute(0,2,3,1).cpu().numpy()+1)/2*255
    vid2_rec_np = (vid2_recon.permute(0,2,3,1).cpu().numpy()+1)/2*255
    vid1_no_np = (vid1_no.permute(0,2,3,1).cpu().numpy()+1)/2*255
    vid1_self_np = (vid1_self.permute(0,2,3,1).cpu().numpy()+1)/2*255
    vid1_cross_np = (vid1_cross.permute(0,2,3,1).cpu().numpy()+1)/2*255
    # vid1_ref_np = (vid1_ref.permute(0,2,3,1).cpu().numpy()+1)/2*255
    ref_img_vis = (ref_img[0].permute(1,2,0).cpu().numpy()+1)/2*255
    vid1_gt_np = np.clip(vid1_gt_np, 0, 255).astype("uint8")
    vid2_gt_np = np.clip(vid2_gt_np, 0, 255).astype("uint8")
    vid1_rec_np = np.clip(vid1_rec_np, 0, 255).astype("uint8")
    vid2_rec_np = np.clip(vid2_rec_np, 0, 255).astype("uint8")
    vid1_no_np = np.clip(vid1_no_np, 0, 255).astype("uint8")
    vid1_self_np = np.clip(vid1_self_np, 0, 255).astype("uint8")
    vid1_cross_np = np.clip(vid1_cross_np, 0, 255).astype("uint8")
    # vid1_ref_np = np.clip(vid1_ref_np, 0, 255).astype("uint8")
    ref_img_vis = np.clip(ref_img_vis, 0, 255).astype("uint8")
    out_name = os.path.join(save_path, f"{gt_id}_vs_{ref_id}_2x4.mp4")
    with imageio.get_writer(out_name, fps=args.fps) as writer:
        for i in range(frames_len):
            row1_col1 = vid1_gt_np[i]
            row1_col2 = vid1_rec_np[i]
            row1_col3 = vid2_gt_np[i]
            row1_col4 = vid2_rec_np[i]
            row2_col1 = ref_img_vis
            row2_col2 = vid1_no_np[i]
            row2_col3 = vid1_self_np[i]
            row2_col4 = vid1_cross_np[i]
            top = np.concatenate([row1_col1, row1_col2, row1_col3, row1_col4], axis=1)
            bottom = np.concatenate([row2_col1, row2_col2, row2_col3, row2_col4], axis=1)
            combined = np.concatenate([top, bottom], axis=0)
            writer.append_data(combined)
    audio_path = os.path.join(gt_latent_dir.replace("cache_latent", "cache_audio"), gt_id + ".wav")
    if os.path.exists(audio_path):
        vclip = mp.VideoFileClip(out_name)
        aclip = mp.AudioFileClip(audio_path)
        final_clip = vclip.set_audio(aclip)
        final_mp4 = os.path.join(save_path, f"{gt_id}_final.mp4")
        final_clip.write_videofile(final_mp4, codec="libx264", audio_codec="aac")
        vclip.close()
        aclip.close()
    break
