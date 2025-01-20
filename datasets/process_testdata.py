import os
import json
import numpy as np
from tqdm import tqdm

stride = 20
motion_length = 64
test_count = 3
cache_path = './BEAT2/cache_latent/'
output_dir = "./datasets/data_json/"
os.makedirs(output_dir, exist_ok=True)

all_files = [f for f in os.listdir(cache_path) if f.endswith('.npz')]
all_files.sort() 

test_files = all_files[:test_count]
train_files = all_files[test_count:]

clips = []

for mode, file_list in zip(['test', 'train'], [test_files, train_files]):
    for file_name in tqdm(file_list, desc=f"Processing {mode} set"):
        file_path = os.path.join(cache_path, file_name)
        
        try:
            motion_data = np.load(file_path, allow_pickle=True)
        except:
            print(f"Cannot load {file_path}")
            continue

        motion = motion_data['random_data']
        total_len = motion.shape[0]

        for i in range(0, total_len - motion_length, stride):
            clip = {
                "video_id": os.path.splitext(file_name)[0],
                "motion_path": file_path,
                "audio_path": file_path.replace("cache_latent", "cache_audio").replace(".npz", ".wav"),
                "mode": mode,
                "start_idx": i,
                "end_idx": i + motion_length
            }
            clips.append(clip)

output_json = os.path.join(output_dir, f"infp_s{stride}_l{motion_length}_test{test_count}.json")
with open(output_json, 'w') as f:
    json.dump(clips, f, indent=4)
