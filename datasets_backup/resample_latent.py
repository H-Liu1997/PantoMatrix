import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

cache_path = './HDTF/cache_latent/'
new_cache_path = './HDTF/cache_latent_resample/'

os.makedirs(new_cache_path, exist_ok=True)

for latent_file in tqdm(os.listdir(cache_path)):
    if not latent_file.endswith(".npz"):
        continue
    
    file_path = os.path.join(cache_path, latent_file)
    data = np.load(file_path, allow_pickle=True)
    motion = data["random_data"]
    
    old_n = motion.shape[0]
    new_n = int(old_n * 30 / 25)
    
    mt = torch.from_numpy(motion).float()
    mt = mt.unsqueeze(0).transpose(1, 2)
    mt = F.interpolate(mt, size=new_n, mode='linear', align_corners=True)
    motion_resampled = mt.transpose(1, 2).squeeze(0).numpy()
    
    new_file_path = os.path.join(new_cache_path, latent_file)
    np.savez(new_file_path, random_data=motion_resampled)
    # print(old_n, new_n, motion_resampled.shape)
