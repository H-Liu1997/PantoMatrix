import random
import os
import numpy as np
import traceback
import torch
from torch.utils.data import IterableDataset
from einops import rearrange, repeat
from glob import glob
import lightning as L
from pathlib import Path
import sys;sys.path.append(str(Path(__file__).parent.parent.parent))


class IterVideoDatasetXC(IterableDataset):
    def __init__(
        self,
        video_dir,
        n_sample_frames,
        key_fliter_npz = [],
        dataset_len = int(1e7),
        global_rank = 0,
        world_size = 1,
        resume_step = 0,
        resolution = [64, 64]
    ):
        super().__init__()
        self.resolution = resolution
        self.resume_step = resume_step
        self.dataset_len = dataset_len
        total_data_list = []
        for item in os.listdir(video_dir):
            select_flag = True
            for key_fliter in key_fliter_npz:
                if key_fliter not in item:
                    select_flag = False
                    break
            if select_flag:
                total_data_list.append(os.path.join(video_dir, item))
        print("total number of npz is", len(total_data_list))
        self.data_list = total_data_list[global_rank::world_size]
        self.n_sample_frames = n_sample_frames
        self.n_sample_latent = (n_sample_frames - 1) // 4 + 1
    
    def __len__(self):
        return self.dataset_len
    
    def spatial_resample(self, x):
        h, w = x.shape[2:]
        if h == self.resolution[0] and w == self.resolution[1]:
            return x
        x = rearrange(x, "c f h w -> f c h w")
        x = torch.nn.functional.interpolate(x, (self.resolution[0], self.resolution[1]), mode="bicubic")
        return rearrange(x, "f c h w -> c f h w")
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        cycle_idx = worker_id * max(len(self.data_list) // num_workers, 1)
        while True:
            data_idx = (cycle_idx + self.resume_step) % len(self.data_list)
            data_path = self.data_list[data_idx]
            cycle_idx += 1
            try:
                data = np.load(data_path)
                ref_image_latents, motion_latents, vid_latents = \
                    data["ref_image_latents"], data["motion_latents"], data["vid_latents"]
                ref_image_latents, motion_latents, vid_latents = \
                    torch.tensor(ref_image_latents, dtype=torch.float32), torch.tensor(motion_latents, dtype=torch.float32), torch.tensor(vid_latents, dtype=torch.float32)

                for ref_image_latent, motion_latent, vid_latent in zip(ref_image_latents, motion_latents, vid_latents):
                    assert ref_image_latent.shape[1] == 1, f"ref_image_latent has bad shape is {ref_image_latent.shape}"
                    assert motion_latent.shape[1] == 1, f"motion_latent has bad shape is {motion_latent.shape}"
                    assert vid_latent.shape[1] > 1, f"vid_latent has bad shape is {vid_latent.shape}"
                    # ref_image_latent = self.spatial_resample(ref_image_latent)
                    # motion_latent = self.spatial_resample(motion_latent)
                    # vid_latent = self.spatial_resample(vid_latent[:, :self.n_sample_latent])
                    sample = {
                        "ref_image_latent": ref_image_latent, # c f h w
                        "motion_latent": motion_latent, # c f h w
                        "vid_latent": vid_latent[:, :self.n_sample_latent], # c f h w
                    }
                    yield sample
            except Exception as e:
                # traceback.print_exc()
                print(data_path, str(e))
                continue
    

class IterStreamVideoDatasetXC(IterableDataset):
    def __init__(
        self,
        video_dir,
        n_sample_frames,
        key_fliter_npz = [],
        dataset_len = int(1e6),
        global_rank = 0,
        world_size = 1,
        resume_step = 0,
    ):
        super().__init__()
        self.resume_step = resume_step
        self.dataset_len = dataset_len
        total_data_list = []
        for item in os.listdir(video_dir):
            select_flag = True
            for key_fliter in key_fliter_npz:
                if key_fliter not in item:
                    select_flag = False
                    break
            if select_flag:
                total_data_list.append(os.path.join(video_dir, item))
        print("total number of npz is", len(total_data_list))
        self.data_list = total_data_list[global_rank::world_size]
        self.n_sample_frames = n_sample_frames
        self.n_sample_latent = (n_sample_frames - 1) // 4 + 1
    
    def __len__(self):
        return self.dataset_len
    
    # Load qingyu 2024.11.20 preprocess data npz
    # NOTE: motion_latent and vid_latent should be swap !!!
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        cycle_idx = 0
        while True:
            data_idx = (cycle_idx + worker_id + self.resume_step) % len(self.data_list)
            data_path = self.data_list[data_idx]
            cycle_idx += 1
            try:
                with np.load(data_path, mmap_mode="r") as data:
                    num_video = data["num_video"]
                    for i in range(num_video):
                        sample = {
                            "ref_image_latent": data[f"ref_image_latent_{i}"], # c f h w
                            "motion_latent": data[f"motion_latent_{i}"], # c f h w
                            "vid_latent": data[f"vid_latent_{i}"][:, :self.n_sample_latent], # c f h w
                        }
                        yield sample
            except Exception as e:
                traceback.print_exc()
                print(data_path, str(e))
                continue