import os
import shutil
import urllib.request
import numpy as np
import torch
import time
import librosa
import soundfile as sf
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image
from decord import VideoReader
from tqdm import tqdm
import imageio
from datasets.face_detector import FaceDetector
from datasets.emo_image import get_mask, scale_bbox
from omegaconf import OmegaConf
from utils import instantiate
import cv2
import subprocess
import tempfile
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--job_id", type=int, default=0)
args = parser.parse_args()

cache_path = "./HDTF/cache_latent_v2"
audio_folder = "./HDTF/cache_audio_v2"
pkl_folder = "./HDTF/cache_facedet_v2"
ori_folder = "./HDTF/cache_ori_v2"
os.makedirs(cache_path, exist_ok=True)
os.makedirs(audio_folder, exist_ok=True)
os.makedirs(pkl_folder, exist_ok=True)
os.makedirs(ori_folder, exist_ok=True)
none_face_list = "./HDTF/none_face_v2.txt"
if not os.path.exists(none_face_list):
    with open(none_face_list, "w") as f:
        pass
cropped_list = "./HDTF/cropped_list_v2.txt"
if not os.path.exists(cropped_list):
    with open(cropped_list, "w") as f:
        pass

face_landmarker_path = "face_landmarker.task"
if not os.path.exists(face_landmarker_path):
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, face_landmarker_path)

detector = FaceDetector(face_landmarker_path, face_detection_confidence=0.5, num_faces=5)

config = OmegaConf.load("/home/weili/haiyang/PantoMatrix/datasets/motion_gen_train.yaml")
motion_encoder = instantiate(config.model.motion_encoder)
params = torch.load(config.model.motion_encoder_path)["state_dict"]
adjusted_dict = {k.replace("motion_encoder.", ""): v for k, v in params.items() if k.startswith("motion_encoder.")}
motion_encoder.load_state_dict(adjusted_dict)
motion_encoder = motion_encoder.to("cuda")


def tensor_to_video(tensor, out_path="output.mp4", fps=25):
    frames = []
    for i in range(tensor.shape[0]):
        frame = tensor[i]
        frame = (frame * 0.5 + 0.5) * 255.0
        frame = frame.clamp(0, 255).permute(1, 2, 0).byte().cpu().numpy()
        frames.append(frame)
    imageio.mimwrite(out_path, frames, fps=fps)
    
def process_video_3bbox(video_path, mouth_bbox_scale=1.4, eye_bbox_scale=1.6, none_face_list=none_face_list):
    import pickle
    # start_time = time.time()
    vr = VideoReader(video_path)
    fps = vr.get_avg_fps()
    frames = vr.get_batch(range(len(vr)))
    frames_tensor = torch.from_numpy(frames.asnumpy()).to('cuda').permute(0, 3, 1, 2).float()
    # print(f"Reading and conversion took {time.time() - start_time:.2f} seconds")
    # start_time = time.time()
    transformed_frames = F.interpolate(frames_tensor, size=(512, 512), mode='bicubic', align_corners=False)
    # print(f"Transforming took {time.time() - start_time:.2f} seconds") 
    frames = transformed_frames.permute(0, 2, 3, 1).cpu().numpy()
    # print(frames.shape)
    
    with open(none_face_list, "a+") as none_face_file:
        out_list = []
        metadata = {}
        for i in range(len(frames)):
            # start_time = time.time()
            f = frames[i].astype(np.uint8)
            res = detector.get_face_xy_rotation_and_keypoints(f, mouth_bbox_scale, eye_bbox_scale)
            # print(f"Processing frame {i} took {time.time() - start_time:.2f} seconds") # 0.01s
            if len(res[8]) == 0:
                none_face_file.write(f"{video_path}\n")
                return {
                    "orig_video": None,
                    "proc_video": None,
                }
            # start_time = time.time()
            face_contour = res[8][0].astype(np.uint8)
            face_contour[face_contour > 0] = 255
            mask = np.zeros((f.shape[0], f.shape[1]), dtype=np.uint8)
            bbox_face = res[6][0]
            x0, y0, x1, y1 = [int(ii) for ii in scale_bbox(bbox_face, f.shape[0], f.shape[1], scale=1.0)]
            mask[y0:y1, x0:x1] = 255
            for eye in ["left_eye", "right_eye"]:
                bbox_eye = res[7][0][eye]
                x0, y0, x1, y1 = [int(ii) for ii in scale_bbox(bbox_eye, f.shape[0], f.shape[1], scale=1.0)]
                mask[y0:y1, x0:x1] = 255
            merged = f * (mask[..., None] / 255.) + face_contour[:, :, :3] * (1 - mask[..., None] / 255.)
            out_list.append(merged.astype(np.uint8))   
            metadata[i] = {
                "mouth": res[6],
                "eyes": res[7],
                "contour": res[8]
            } 
            
    # crop the center of the face 
    fc = np.zeros(2)
    for i in range(len(out_list)):
        cnt = metadata[i]["contour"][0]
        pts = cnt.reshape(-1, cnt.shape[-1]) if cnt.ndim == 3 else cnt
        fc += np.mean(pts[:, :2], axis=0)
    fc /= len(out_list)
    H, W = out_list[0].shape[:2]
    fc[0] *= W
    fc[1] *= H
    # print(fc)
    center = np.array([W / 2, H / 2])
    allowed = np.array([0.1 * W, 0.1 * H])
    if np.all(np.abs(fc - center) <= allowed):
        return {
        "orig_video": frames.astype(np.uint8),
        "proc_video": np.stack(out_list),
        "fps": fps,
        }
    else:
        m = int(min(fc[0], W - fc[0], fc[1], H - fc[1]))
        x0, y0 = int(fc[0] - m), int(fc[1] - m)
        crop_size = m * 2
        proc_video = np.array([cv2.resize(frame[y0:y0+crop_size, x0:x0+crop_size], (512, 512), interpolation=cv2.INTER_CUBIC) for frame in out_list])
        orig_video = np.array([cv2.resize(frame[y0:y0+crop_size, x0:x0+crop_size], (512, 512), interpolation=cv2.INTER_CUBIC) for frame in frames_np])
        scale_factor = 512 / crop_size
        print(f"Video {video_id} cropped: original {W}x{H}, crop region {crop_size}x{crop_size}, scale factor {scale_factor:.2f}")
        with open(cropped_list, "a+") as cropped_file:
            cropped_file.write(f"{video_path}\n")
        return {
            "orig_video": orig_video,
            "proc_video": proc_video,
            "fps": fps,
        }
        

def adjust_fps_ffmpeg(video_np, output_video, source_fps=25, target_fps=25):
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_in:
        temp_in_path = temp_in.name
    imageio.mimwrite(temp_in_path, video_np, fps=source_fps, quality=8)
    cmd = ["ffmpeg", "-y", "-i", temp_in_path, "-vf", f"fps={target_fps}", output_video]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        os.remove(temp_in_path)
        raise RuntimeError(proc.stderr.decode())
    print(f"Adjusted FPS from {source_fps} to {target_fps}, saved as {output_video}")
    os.remove(temp_in_path)
    
    
# def adjust_fps_torch_interp(video_np, output_video, source_fps=25, target_fps=25):
#     T, H, W, C = video_np.shape
#     if source_fps == target_fps:
#         imageio.mimwrite(output_video, video_np, fps=target_fps, quality=5)
#         # print(f"FPS unchanged, directly saved as {output_video}")
#         return
#     new_T = int(round(T * target_fps / source_fps))
#     video_tensor = torch.from_numpy(video_np).float().cuda()
#     video_tensor = video_tensor.permute(1, 2, 3, 0).reshape(H*W, C, T)
#     interp_video = torch.nn.functional.interpolate(video_tensor, scale_factor=target_fps/source_fps, mode='linear', align_corners=False)
#     # print("before", video_tensor.shape, "after", interp_video.shape)
#     interp_video = interp_video.reshape(H, W, C, new_T).permute(3, 0, 1, 2)
#     video_out = interp_video.cpu().numpy().astype(np.uint8).clip(0, 255)
#     imageio.mimwrite(output_video, video_out, fps=target_fps, quality=5)
#     print(f"Adjusted FPS from {source_fps} to {target_fps} using time interpolation, saved as {output_video}")

    
def get_motion_latent(video_path):
    start_time = time.time()
    outputs = process_video_3bbox(video_path)
    frames_np = outputs["proc_video"]
    frames_orig = outputs["orig_video"]
    # print(f"Processing video {video_path} took {time.time() - start_time:.2f} seconds")
    if frames_np is None: 
        return None
    video_id = os.path.basename(video_path)[:-4]
    test_out_video = os.path.join(pkl_folder, f"{video_id}.mp4")
    orig_out_video = os.path.join(ori_folder, f"{video_id}.mp4")
    
    # start_time = time.time()
    adjust_fps_ffmpeg(frames_np.clip(0, 255), test_out_video, source_fps=outputs["fps"], target_fps=25)
    adjust_fps_ffmpeg(frames_orig.clip(0, 255), orig_out_video, source_fps=outputs["fps"], target_fps=25)
    # imageio.mimwrite(test_out_video, frames_np.clip(0, 255), fps=25)
    # imageio.mimwrite(orig_out_video, frames_orig.clip(0, 255), fps=25)
    # print(f"Converting tensor to video took {time.time() - start_time:.2f} seconds")
    
    # start_time = time.time()
    frames_np = frames_np.astype(np.float32) / 255.
    frames_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2).to("cuda")
    frames_tensor = (frames_tensor - 0.5) / 0.5
    # print(f"Vectorized transform took {time.time() - start_time:.2f} seconds")

    chunk_size = 250
    all_latents = []
    with torch.no_grad():
        for i in range(0, frames_tensor.shape[0], chunk_size):
            batch = frames_tensor[i:i+chunk_size]
            latents_chunk = motion_encoder(batch)
            all_latents.append(latents_chunk[0].detach().cpu().numpy())
    # print(f"Motion encoding took {time.time() - start_time:.2f} seconds")
    return np.concatenate(all_latents, axis=0)


src_folder = "/mnt/weka/training_data_1/hdtf_full/videos_resampled"
all_list = sorted([x for x in os.listdir(src_folder) if x.endswith(".mp4")])
half_lens = len(all_list)//6
all_list = all_list[args.job_id * half_lens: (args.job_id + 1) * half_lens]
finished = os.listdir(pkl_folder)
all_none_face = []
with open(none_face_list, "r") as f:
    for line in f:
        all_none_face.append(os.path.basename(line.strip()))
# print(all_none_face)

for data_file in tqdm(all_list):
    if not data_file.endswith(".mp4"): 
        continue
    if data_file in finished:
        continue
    if data_file in all_none_face:
        continue
    video_id = os.path.splitext(data_file)[0]
    latent_np = get_motion_latent(os.path.join(src_folder, data_file))
    if latent_np is None:
        continue
    np.savez(os.path.join(cache_path, f"{video_id}.npz"), random_data=latent_np)
    wav_path = os.path.join(src_folder, video_id + ".wav")
    if os.path.exists(wav_path):
        shutil.copy(wav_path, os.path.join(audio_folder, f"{video_id}.wav"))

# single test case # /home/weili/haiyang/WDA_DavidCicilline_000_002.mp4 /home/weili/haiyang/WDA_DavidCicilline_000_002.wav, and do tensor_to_video check, save in /home/weili/haiyang/visualization.mp4
# test_video = "/home/weili/haiyang/WDA_DavidCicilline_000_002.mp4"
# test_audio = "/home/weili/haiyang/WDA_DavidCicilline_000_002.wav"
# test_out_video = "/home/weili/haiyang/visualization.mp4"
# vid_tensor = process_video_3bbox(test_video)
# if vid_tensor is not None:
#     print(vid_tensor.shape)
#     tensor_to_video(vid_tensor, out_path=test_out_video, fps=25)
#     frames_tensor = vid_tensor.to("cuda:2")
#     with torch.no_grad():
#         latents = motion_encoder(frames_tensor)
#     print(latents[0].shape) 