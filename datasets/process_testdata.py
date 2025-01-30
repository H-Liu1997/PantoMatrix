import os
import json
import numpy as np
from tqdm import tqdm
import librosa

stride = 20
motion_length = 64
k_wild = 2
n_audio_only = 2
cache_path = './HDTF/cache_latent/'
output_dir = "./datasets/data_json/"
os.makedirs(output_dir, exist_ok=True)

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

test_wild_persons = ["RD_Radio10", "WDA_AlexandriaOcasioCortez"]
for p in test_wild_persons:
    print(f"{p} => test_wild")

test_wild_files = []
for p in test_wild_persons:
    for b in data_dict[p]:
        test_wild_files += data_dict[p][b]

remaining_persons = list(set(persons) - set(test_wild_persons))
remaining_persons.sort()
print(remaining_persons)
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
    for file_name in tqdm(fl, desc=mode):
        fp = os.path.join(cache_path, file_name)
        try:
            md = np.load(fp, allow_pickle=True)
        except:
            continue
        motion = md['random_data']
        total_len = motion.shape[0]
        if total_len <= motion_length: 
            continue
        ml = total_len / 30
        ap = fp.replace("cache_latent", "cache_audio").replace(".npz", ".wav")
        audio, sr = librosa.load(ap, sr=16000)
        al = audio.shape[0]/sr
        if abs(ml - al) > 0.01:
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

out_json = os.path.join(output_dir, f"infp_s{stride}_l{motion_length}_kw{k_wild}_na{n_audio_only}.json")
with open(out_json, 'w') as f:
    json.dump(clips, f)
