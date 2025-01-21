import os
import shutil
import urllib.request
import numpy as np
import torch
import librosa
import soundfile as sf
import torchvision.transforms as T
from PIL import Image
from decord import VideoReader
from tqdm import tqdm
import imageio
from datasets.face_detector import FaceDetector
from datasets.emo_image import get_mask
from omegaconf import OmegaConf
from utils import instantiate

cache_path = "./HDTF/cache_latent"
audio_folder = "./HDTF/cache_audio"
pkl_folder = "./HDTF/cache_facedet"
os.makedirs(cache_path, exist_ok=True)
os.makedirs(audio_folder, exist_ok=True)
os.makedirs(pkl_folder, exist_ok=True)

face_landmarker_path = "face_landmarker.task"
if not os.path.exists(face_landmarker_path):
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, face_landmarker_path)

detector = FaceDetector(face_landmarker_path, face_detection_confidence=0.5, num_faces=5)
transform = T.Compose([
    T.Resize((512, 512), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize([0.5], [0.5])
])

def tensor_to_video(tensor, out_path="output.mp4", fps=25):
    frames = []
    for i in range(tensor.shape[0]):
        frame = tensor[i]
        frame = (frame * 0.5 + 0.5) * 255.0
        frame = frame.clamp(0, 255).permute(1, 2, 0).byte().cpu().numpy()
        frames.append(frame)
    imageio.mimwrite(out_path, frames, fps=fps)
    
def process_video_3bbox(video_path, mouth_bbox_scale=1.4, eye_bbox_scale=1.6):
    import pickle
    vr = VideoReader(video_path)
    frames = vr.get_batch(range(len(vr))).asnumpy()
    out_list = []
    metadata = {}
    for i in range(len(frames)):
        f = frames[i].astype(np.uint8)
        res = detector.get_face_xy_rotation_and_keypoints(f, mouth_bbox_scale, eye_bbox_scale)
        if len(res[5]) == 0:
            continue
        face_contour = res[8][0].astype(np.uint8)
        face_contour[face_contour > 0] = 255
        mask1 = get_mask([res[6][0][0], res[6][0][1], res[6][0][2], res[6][0][3]], f.shape[0], f.shape[1], 1.0, return_pil=False)
        le = res[7][0]["left_eye"]
        re = res[7][0]["right_eye"]
        mask2a = get_mask([le[0], le[1], le[2], le[3]], f.shape[0], f.shape[1], 1.0, return_pil=False)
        mask2b = get_mask([re[0], re[1], re[2], re[3]], f.shape[0], f.shape[1], 1.0, return_pil=False)
        mask = np.maximum(mask1, (mask2a + mask2b).clip(0, 255)).astype(np.uint8)
        merged = f * (mask / 255.) + face_contour[:, :, :3] * (1 - mask / 255.)
        img = Image.fromarray(merged.astype(np.uint8))
        out_list.append(transform(img).unsqueeze(0))
        # metadata[i] = {
        #     "mouth": res[6],
        #     "eyes": res[7],
        #     "contour": res[8]
        # }   
    # video_id = os.path.basename(video_path)[:-4]
    # print(os.path.join(pkl_folder, f"{video_id}.pkl"))
    # with open(os.path.join(pkl_folder, f"{video_id}.pkl"), "wb") as f:
    #     pickle.dump(metadata, f)
    if not out_list:
        return None
    return torch.cat(out_list, dim=0)


config = OmegaConf.load("/home/weili/haiyang/PantoMatrix/datasets/motion_gen_train.yaml")
motion_encoder = instantiate(config.model.motion_encoder)
params = torch.load(config.model.motion_encoder_path)["state_dict"]
adjusted_dict = {k.replace("motion_encoder.", ""): v for k, v in params.items() if k.startswith("motion_encoder.")}
motion_encoder.load_state_dict(adjusted_dict)
motion_encoder = motion_encoder.to("cuda:2")

def get_motion_latent(video_path):
    frames_tensor = process_video_3bbox(video_path)
    if frames_tensor is None: 
        return None
    video_id = os.path.basename(video_path)[:-4]
    test_out_video = os.path.join(pkl_folder, f"{video_id}.mp4")
    tensor_to_video(frames_tensor, out_path=test_out_video, fps=25)
    frames_tensor = frames_tensor.to("cuda:2")
    chunk_size = 250
    all_latents = []
    with torch.no_grad():
        for i in range(0, frames_tensor.shape[0], chunk_size):
            batch = frames_tensor[i:i+chunk_size]
            latents_chunk = motion_encoder(batch)
            all_latents.append(latents_chunk[0].detach().cpu().numpy())
    return np.concatenate(all_latents, axis=0)

src_folder = "/mnt/weka/training_data_1/hdtf_full/videos_resampled"
all_list = sorted([x for x in os.listdir(src_folder) if x.endswith(".mp4")])
half_lens = len(all_list)//2
all_list = all_list[:half_lens]
finished = os.listdir("/home/weili/haiyang/PantoMatrix/HDTF/cache_facedet")

for data_file in tqdm(all_list):
    if not data_file.endswith(".mp4"): 
        continue
    if data_file in finished:
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