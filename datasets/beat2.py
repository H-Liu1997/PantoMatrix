import json
import torch
from torch.utils import data
import numpy as np
import librosa
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from emage_utils.motion_io import beat_format_load, MASK_DICT

class BEAT2Dataset(data.Dataset):
    def __init__(self, cfg, split):
        vid_meta = []
        for data_meta_path in cfg.data.meta_paths:
            vid_meta.extend(json.load(open(data_meta_path, "r")))
        self.vid_meta = [item for item in vid_meta if item.get("mode") == split]
        self.mean = 0
        self.std = 1
        self.joint_mask = MASK_DICT[cfg.model.joint_mask] if cfg.model.joint_mask is not None else None
        self.data_list = self.vid_meta
        self.fps = cfg.model.pose_fps
        self.audio_sr = cfg.model.audio_sr

    def __len__(self):
        return len(self.data_list)
    
    @staticmethod
    def normalize(motion, mean, std):
        return (motion - mean) / (std + 1e-7)
    
    @staticmethod
    def inverse_normalize(motion, mean, std):
        # return motion * torch.from_numpy(std).to(motion.device) + torch.from_numpy(mean).to(motion.device)
        return motion * torch.tensor(std).to(motion.device) + torch.tensor(mean).to(motion.device)

    def __getitem__(self, item):
        data_item = self.data_list[item]
        smplx_data = beat_format_load(data_item["motion_path"], mask=self.joint_mask)
        sdx, edx = data_item["start_idx"], data_item["end_idx"]
        motion = smplx_data["poses"][sdx:edx]
        SMPLX_FPS = 30
        downsample_factor = SMPLX_FPS // self.fps
        motion = motion[::downsample_factor]
        # motion = self.normalize(motion, self.mean, self.std)
        
        audio, _ = librosa.load(data_item["audio_path"], sr=self.audio_sr)
        sdx_audio = sdx * int((1 / SMPLX_FPS) * self.audio_sr)
        edx_audio = edx * int((1 / SMPLX_FPS) * self.audio_sr)
        audio = audio[sdx_audio:edx_audio]
             
        motion_tensor = torch.from_numpy(motion).float()
        audio_tensor = torch.from_numpy(audio).float()
       
        return dict(
            motion=motion_tensor,
            audio=audio_tensor, 
        )

class BEAT2DatasetEamge(BEAT2Dataset):
    def __init__(self, cfg, split):
        super().__init__(cfg, split)

    def __getitem__(self, item):
        data_item = self.data_list[item]
        smplx_data = beat_format_load(data_item["motion_path"], mask=None)
        sdx, edx = data_item["start_idx"], data_item["end_idx"]
        motion = smplx_data["poses"][sdx:edx]
        expressions = smplx_data["expressions"][sdx:edx]
        trans = smplx_data["trans"][sdx:edx]
        SMPLX_FPS = 30
        downsample_factor = SMPLX_FPS // self.fps
        motion = motion[::downsample_factor]
        motion = self.normalize(motion, self.mean, self.std)
        
        audio, _ = librosa.load(data_item["audio_path"], sr=self.audio_sr)
        sdx_audio = sdx * int((1 / SMPLX_FPS) * self.audio_sr)
        edx_audio = edx * int((1 / SMPLX_FPS) * self.audio_sr)
        audio = audio[sdx_audio:edx_audio]
             
        motion_tensor = torch.from_numpy(motion).float()
        audio_tensor = torch.from_numpy(audio).float()
        expressions_tesnor = torch.from_numpy(expressions).float()
        trans_tensor = torch.from_numpy(trans).float()

        return dict(
            motion=motion_tensor,
            audio=audio_tensor, 
            expressions=expressions_tesnor,
            trans=trans_tensor,
        )

class BEAT2DatasetEamgeFootContact(BEAT2Dataset):
    def __init__(self, cfg, split):
        super().__init__(cfg, split)
        self.stat = np.load("/home/weili/haiyang/PantoMatrix/HDTF/global_stats.npz", allow_pickle=True)
        self.mean = 0.0
        self.std = 0.03352

    def __getitem__(self, item):
        data_item = self.data_list[item]
        motion_dict = np.load(data_item["motion_path"], allow_pickle=True)
        sdx, edx = data_item["start_idx"], data_item["end_idx"]
        # print(motion_dict["random_data"].shape, sdx, edx)
        motion = motion_dict["random_data"][sdx:edx]
        # motion = self.normalize(motion, self.mean, self.std)
        
        length = data_item["frames"] - (edx-sdx) - 1
        ref_sdx = np.random.randint(0, length)
        ref_edx = ref_sdx + (edx-sdx)
        ref_motion = motion_dict["random_data"][ref_sdx:ref_edx]
        
        SMPLX_FPS = 30
        audio, _ = librosa.load(data_item["audio_path"], sr=self.audio_sr)
        sdx_audio = sdx * int((1 / SMPLX_FPS) * self.audio_sr)
        edx_audio = edx * int((1 / SMPLX_FPS) * self.audio_sr)
        audio = audio[sdx_audio:edx_audio]
             
        motion_tensor = torch.from_numpy(motion).float()
        audio_tensor = torch.from_numpy(audio).float()
        ref_motion_tensor = torch.from_numpy(ref_motion).float()
        # print(motion_tensor.shape[0]/30, audio_tensor.shape[0]/16000)

        return dict(
            motion_latent=motion_tensor,
            audio=audio_tensor, 
            style_latent=ref_motion_tensor,
        )
        
        
class BEAT2DatasetEamgeFix(BEAT2Dataset):
    def __init__(self, cfg, split):
        super().__init__(cfg, split)
        self.stat = np.load("/home/weili/haiyang/PantoMatrix/HDTF/global_stats.npz", allow_pickle=True)
        self.mean = 0.0
        self.std = 0.03352

    def __getitem__(self, item):
        data_item = self.data_list[item]
        motion_dict = np.load(data_item["motion_path"], allow_pickle=True)
        sdx, edx = data_item["start_idx"], data_item["end_idx"]
        # print(motion_dict["random_data"].shape, sdx, edx)
        motion = motion_dict["random_data"][sdx:edx]
        # motion = self.normalize(motion, self.mean, self.std)
        
        # length = data_item["frames"] - (edx-sdx) - 1
        # ref_sdx = np.random.randint(0, length)
        # ref_edx = ref_sdx + (edx-sdx)
        ref_motion = motion # motion_dict["random_data"][ref_sdx:ref_edx]
        
        SMPLX_FPS = 30
        audio, _ = librosa.load(data_item["audio_path"], sr=self.audio_sr)
        sdx_audio = sdx * int((1 / SMPLX_FPS) * self.audio_sr)
        edx_audio = edx * int((1 / SMPLX_FPS) * self.audio_sr)
        audio = audio[sdx_audio:edx_audio]
             
        motion_tensor = torch.from_numpy(motion).float()
        audio_tensor = torch.from_numpy(audio).float()
        ref_motion_tensor = torch.from_numpy(ref_motion).float()
        # print(motion_tensor.shape[0]/30, audio_tensor.shape[0]/16000)

        return dict(
            motion_latent=motion_tensor,
            audio=audio_tensor, 
            style_latent=ref_motion_tensor,
        )
        
        
class BEAT2DatasetEamgeRandom(BEAT2Dataset):
    def __init__(self, cfg, split):
        super().__init__(cfg, split)
        self.stat = np.load("/home/weili/haiyang/PantoMatrix/HDTF/global_stats.npz", allow_pickle=True)
        self.mean = 0.0
        self.std = 0.03352

    def __getitem__(self, item):
        data_item = self.data_list[item]
        motion_dict = np.load(data_item["motion_path"], allow_pickle=True)
        sdx, edx = data_item["start_idx"], data_item["end_idx"]
        # print(motion_dict["random_data"].shape, sdx, edx)
        motion = motion_dict["random_data"][sdx:edx]
        # motion = self.normalize(motion, self.mean, self.std)
        
        if np.random.rand() > self.cfg.random_mix:
            length = data_item["frames"] - (edx-sdx) - 1
            ref_sdx = np.random.randint(0, length)
            ref_edx = ref_sdx + (edx-sdx)
            ref_motion = motion_dict["random_data"][ref_sdx:ref_edx]
        else:
            ref_motion = motion
        
        SMPLX_FPS = 30
        audio, _ = librosa.load(data_item["audio_path"], sr=self.audio_sr)
        sdx_audio = sdx * int((1 / SMPLX_FPS) * self.audio_sr)
        edx_audio = edx * int((1 / SMPLX_FPS) * self.audio_sr)
        audio = audio[sdx_audio:edx_audio]
             
        motion_tensor = torch.from_numpy(motion).float()
        audio_tensor = torch.from_numpy(audio).float()
        ref_motion_tensor = torch.from_numpy(ref_motion).float()
        # print(motion_tensor.shape[0]/30, audio_tensor.shape[0]/16000)

        return dict(
            motion_latent=motion_tensor,
            audio=audio_tensor, 
            style_latent=ref_motion_tensor,
        )
