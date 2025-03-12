import os
import json
import numpy as np
from tqdm import tqdm
import librosa
import cv2

def get_origin_video_path(video_metadata):
    pass
    
stride = 20
motion_length = 64
k_wild = 4
output_dir = "./datasets/data_json/"
os.makedirs(output_dir, exist_ok=True)
final_npz_path = '/mnt/weka/training_data_xc/Dyadic_Forbes_TwoFace_HQ300_835.npz'

final_npz_list = np.load(final_npz_path, allow_pickle=True)["arr_0"].tolist()
total_lens = len(final_npz_list)
"""
exmaple:
[['/mnt/weka/training_data_xc/Forbes_processed', '-0Y535vZbAQ-Scene-003'],
 ['/mnt/weka/training_data_xc/Forbes_processed', '-0Y535vZbAQ-Scene-004'],
 ['/mnt/weka/training_data_xc/Forbes_processed', '-0Y535vZbAQ-Scene-047'],
 ['/mnt/weka/training_data_xc/Forbes_processed', '-J4aGdkBvRE-Scene-003'],
"""
# get the raw_video_name, such as '-0Y535vZbAQ'
raw_video_names = [x[1].split('-Scene')[0] for x in final_npz_list]
raw_video_names = list(set(raw_video_names))

test_wild_names = raw_video_names[:k_wild]
train_names = raw_video_names[k_wild:]

clips = []
for train_name in tqdm(train_names):
    sample_with_name = [x for x in final_npz_list if train_name in x[1]]
    for sample in sample_with_name:
        data_path, file_name = sample
        meta_data_path = os.path.join(data_path, "metadata", file_name, "metadata.npz")
        video_metadata = np.load(meta_data_path, allow_pickle=True)["arr_0"].tolist()
        # print(video_metadata)
        resampled_video_path = os.path.join(data_path, "videos_resampled", video_metadata["path_to_video"])
        origin_video_path = None # get_origin_video_path(video_metadata)
        total_len = video_metadata["frame_count"]

        audio_path = resampled_video_path.replace("+resampled.mp4", "+audio_full.wav") #origin_video_path.replace(".mp4", ".wav")
        audio_self_path = audio_path.replace("+audio_full.wav", "+audio_full_0.wav")
        audio_other_path = audio_path.replace("+audio_full.wav", "+audio_full_1.wav")
        if not os.path.exists(audio_path):
            print(audio_path, "not exists")
            continue
        if not os.path.exists(audio_self_path):
            print(audio_self_path, "not exists")
            continue
        if not os.path.exists(audio_other_path):
            print(audio_other_path, "not exists")
            continue
        
        for i in range(0, total_len - motion_length, stride):
            clips.append({
                "video_id": file_name,
                "origin_video_path": origin_video_path,
                "resampled_video_path": resampled_video_path,
                "audio_path": audio_path,
                "audio_self_path": audio_self_path,
                "audio_other_path": audio_other_path,
                "mode": "train",
                "start_idx": i,
                "end_idx": i + motion_length,
                "frames": total_len, 
                "metadata_path": meta_data_path,
            })
            break
print("train clips:", len(clips))
            
for test_wild_name in tqdm(test_wild_names):
    sample_with_name = [x for x in final_npz_list if test_wild_name in x[1]][0:1]
    for sample in sample_with_name:
        data_path, file_name = sample
        meta_data_path = os.path.join(data_path, "metadata", file_name, "metadata.npz")
        video_metadata = np.load(meta_data_path, allow_pickle=True)["arr_0"].tolist()
        resampled_video_path = os.path.join(data_path, "videos_resampled", video_metadata["path_to_video"])
        origin_video_path = None # get_origin_video_path(video_metadata)
        total_len = video_metadata["frame_count"]

        audio_path = resampled_video_path.replace("+resampled.mp4", "+audio_full.wav") #origin_video_path.replace(".mp4", ".wav")
        audio_self_path = audio_path.replace("+audio_full.wav", "+audio_full_0.wav")
        audio_other_path = audio_path.replace("+audio_full.wav", "+audio_full_1.wav")
        if not os.path.exists(audio_path):
            print(audio_path, "not exists")
            continue
        if not os.path.exists(audio_self_path):
            print(audio_self_path, "not exists")
            continue
        if not os.path.exists(audio_other_path):
            print(audio_other_path, "not exists")
            continue
        
        for i in range(0, total_len - motion_length, stride):
            clips.append({
                "video_id": file_name,
                "origin_video_path": origin_video_path,
                "resampled_video_path": resampled_video_path,
                "audio_path": audio_path,
                "audio_self_path": audio_self_path,
                "audio_other_path": audio_other_path,
                "mode": "test_wild",
                "start_idx": i,
                "end_idx": i + motion_length,
                "frames": total_len, 
                "metadata_path": meta_data_path,
            })
            break
print("all clips:", len(clips))

out_json = os.path.join(output_dir, f"dyanic_cache_s{stride}_l{motion_length}_v1.json")
with open(out_json, 'w') as f:
    json.dump(clips, f)