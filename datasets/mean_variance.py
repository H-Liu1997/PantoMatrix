import os
import numpy as np
from tqdm import tqdm

cache_path = './HDTF/cache_latent/'

global_mean = None
global_variance = None
global_n = 0

for latent_file in tqdm(os.listdir(cache_path)):
    if not latent_file.endswith(".npz"):
        continue
    
    file_path = os.path.join(cache_path, latent_file)
    data = np.load(file_path, allow_pickle=True)
    motion = data["random_data"]  # n, d where d = 512
    
    local_n = motion.shape[0]
    local_mean = np.mean(motion, axis=0)
    local_variance = np.var(motion, axis=0)
    
    if global_mean is None:
        global_mean = local_mean
        global_variance = local_variance
        global_n = local_n
    else:
        delta = local_mean - global_mean
        total_n = global_n + local_n
        global_mean += delta * local_n / total_n
        global_variance = (
            global_variance * global_n +
            local_variance * local_n +
            delta ** 2 * global_n * local_n / total_n
        ) / total_n
        global_n = total_n

np.savez('./HDTF/global_stats.npz', mean=global_mean, variance=global_variance)
print(global_mean, global_variance)