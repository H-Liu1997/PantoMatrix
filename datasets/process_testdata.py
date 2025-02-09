import os
import json
import numpy as np
from tqdm import tqdm
import librosa
import cv2

stride = 20
motion_length = 64
k_wild = 2
n_audio_only = 2
cache_path = './HDTF/cache_latent_v4/'
output_dir = "./datasets/data_json/"
os.makedirs(output_dir, exist_ok=True)
fps = 24
meta_path = '/mnt/weka/training_data_1/hdtf_full/metadata'
root_path = '/mnt/weka/training_data_1/hdtf_full/videos_resampled'


def bbox_in_center(video_id):
    meta_data = np.load(os.path.join(meta_path, video_id, 'metadata.npz'), allow_pickle=True)
    bbox_data = meta_data['arr_0'].item()  # Convert to dictionary
   
    # Extract bounding box
    frame_data = bbox_data.get('frame_data', {})
    bounding_boxes = frame_data.get('bounding_box', {})

    # Get the first bounding box (assuming we use frame 0)
    if 0 in bounding_boxes:
        bbox = bounding_boxes[0]  # This should be a NumPy array
        if bbox.shape[0] > 0:  # Ensure it's non-empty
            # Compute center for the first bounding box entry
            centerx = bbox[0][0] + (bbox[0][2] - bbox[0][0]) / 2
            centery = bbox[0][1] + (bbox[0][3] - bbox[0][1]) / 2
            # print(f"Center: ({centerx}, {centery})")
     
    # get h, w
    ori_video = os.path.join(root_path, video_id + '.mp4')
    cap = cv2.VideoCapture(ori_video)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # print(f"Width: {w}, Height: {h}")
    relativex = centerx / w
    relativey = centery / h
    # print(f"Relative: ({relativex}, {relativey})")
    if abs(relativex - 0.5) > 0.10 or abs(relativey - 0.5) > 0.10:
        print(f"Warning: Center not in the middle for {video_id}")
        return False
    else:
        return True
    

def parse_name(fname):
    base = fname[:-4]
    parts = base.split('_')
    return '_'.join(parts[:2]), '_'.join(parts[2:])

file_list = [f for f in os.listdir(cache_path) if f.endswith('.npz')]
file_list.sort()
data_dict = {}
for f in file_list:
    p, b = parse_name(f)
    if p not in data_dict:
        data_dict[p] = {}
    if b not in data_dict[p]:
        data_dict[p][b] = []
    data_dict[p][b].append(f)

persons = sorted(data_dict.keys())
print(persons)

to_remove = []
for p in tqdm(persons):
    file_name = data_dict[p].values()
    # print(f"{p} => {file_name}")
    file_list = list(file_name)[0]  # Convert dict_values to a list and get the first item
    video_id = file_list[0][:-4]  # Extract filename and remove ".npz"
    # print(video_id)
    if not bbox_in_center(video_id):
        to_remove.append(p)
print(f"Removing {len(to_remove)} persons")

test_wild_persons = ['RD_Radio37', 'WDA_NancyPelosi1', 'WDA_JoaquinCastro', 'WRA_JonKyl', 'WDA_LloydDoggett1', 'WRA_RoyBlunt', 'WDA_DebbieWassermanSchultz', 'WRA_MitchDaniels1', 'WRA_CarlyFiorina0', 'WDA_ByronDorgan1', 'WRA_KevinBrady2', 'RD_Radio42', 'RD_Radio14', 'WRA_SteveScalise1', 'WRA_KayBaileyHutchison', 'WDA_TerriSewell', 'RD_Radio35', 'WDA_KathyCastor1', 'WDA_ChrisVanHollen1', 'WRA_DeanHeller']
# test_wild_persons = ["RD_Radio10", "WDA_AlexandriaOcasioCortez"]
for p in test_wild_persons:
    print(f"{p} => test_wild")

test_wild_files = []
for p in test_wild_persons:
    for b in data_dict[p]:
        test_wild_files += data_dict[p][b]

remaining_persons = list(set(persons) - set(test_wild_persons) - set(to_remove))
remaining_persons.sort()
print(len(remaining_persons))
test_audio_only_candidates = []
for p in remaining_persons:
    bigvideos = data_dict[p].keys()
    if len(bigvideos) >= 10:
        test_audio_only_candidates.append(p)

test_audio_only_persons = test_audio_only_candidates[:n_audio_only]
for p in test_audio_only_persons:
    print(f"{p} => test_audio_only")

train_persons = list(set(remaining_persons) - set(test_audio_only_persons))
test_audio_only_files = []
test_trained_files = []
train_files = []
train_persons.sort()
for p in test_audio_only_persons:
    bigvideos = sorted(data_dict[p].keys())
    # print(f"{p} => {len(bigvideos)} bigvideos: 2 test_audio_only, 2 test_trained, rest train")
    test_audio_only_bv = bigvideos[:4]
    test_trained_bv = bigvideos[4:8]
    for b in test_audio_only_bv:
        test_audio_only_files += data_dict[p][b]
    for b in test_trained_bv:
        test_trained_files += data_dict[p][b]
    for b in bigvideos[4:]:
        train_files += data_dict[p][b]

print(f"test_audio_only_persons: {len(test_audio_only_persons)}")
print(test_audio_only_persons)
print(f"train_persons: {len(train_persons)}")
print(train_persons)
for p in train_persons:
    bigvideos = sorted(data_dict[p].keys())
    for b in bigvideos:
        train_files += data_dict[p][b]

clips = []
def process_files(fl, mode):
    counter = 0
    for file_name in tqdm(fl, desc=mode):
        fp = os.path.join(cache_path, file_name)
        try:
            md = np.load(fp, allow_pickle=True)
        except:
            print(f"Error loading {fp}")
            continue
        motion = md['random_data']
        total_len = motion.shape[0]
        if total_len <= motion_length:
            print(f"Motion too short: {total_len}") 
            continue
        ml = total_len / fps
        ap = fp.replace("cache_latent", "cache_audio").replace(".npz", ".wav")
        try:
            audio, sr = librosa.load(ap, sr=16000)
        except:
            print(f"Error loading {ap}")
            continue
        al = audio.shape[0]/sr
        if abs(ml - al) > 0.02:
            counter += 1
            print(f"Length mismatch: {ml} vs {al}, {counter}")
            continue
        for i in range(0, total_len - motion_length, stride):
            clips.append({
                "video_id": file_name[:-4],
                "motion_path": fp,
                "audio_path": ap,
                "mode": mode,
                "start_idx": i,
                "end_idx": i + motion_length,
                "frames": total_len,
            })

process_files(test_wild_files, "test_wild")
process_files(test_audio_only_files, "test_audio_only")
process_files(test_trained_files, "test_trained")
process_files(train_files, "train")

out_json = os.path.join(output_dir, f"infp_s{stride}_l{motion_length}_kw{k_wild}_na{n_audio_only}_v4.json")
with open(out_json, 'w') as f:
    json.dump(clips, f)
