import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from detectors import S3FD
from SyncNetInstance import SyncNetInstance
from run_pipeline import syncnet_pipeline
from run_syncnet import syncnet_run
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class SyncNetDataset(Dataset):
    def __init__(self, video_list):
        self.video_list = video_list

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        video_path = self.video_list[index]
        detector = S3FD(device="cuda")
        syncnet = SyncNetInstance()
        syncnet.loadParameters(SyncNetDataset.syncnet_model_path)

        vid_name = os.path.basename(video_path)
        dir_name = os.path.dirname(video_path)
        tmp_dir = os.path.join(dir_name, vid_name[:-4])
        os.makedirs(tmp_dir, exist_ok=True)

        syncnet_pipeline(detector, video_path, vid_name, data_dir=tmp_dir)
        _, conf, _, _, frames = syncnet_run(syncnet, video_path, vid_name, data_dir=tmp_dir)

        os.system(f"rm -rf {tmp_dir}")  # Clean up temp files

        return conf.item(), len(frames), vid_name

# Path to SyncNet model
SyncNetDataset.syncnet_model_path = "/home/weili/haiyang/PantoMatrix/metric/model_weight/syncnet_v2.model"
    
def eval_syncnet_videos(video_path_list, output_json, num_workers=4):
    dataset = SyncNetDataset(video_path_list)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=num_workers, collate_fn=lambda x: x[0], drop_last=False, shuffle=False)

    results = []
    total_conf, total_len = 0, 0

    for conf, ln, vid in tqdm(dataloader, total=len(video_path_list)):
        results.append({"video": vid, "conf": conf, "length": ln})
        total_conf += conf * ln
        total_len += ln

    avg_conf = total_conf / total_len if total_len else 0
    with open(output_json, "w") as f:
        json.dump(results, f)
    return avg_conf

if __name__ == "__main__":
    gt_path = "/home/weili/haiyang/PantoMatrix/HDTF/cache_merge_v4"
    video_pred_path = "/home/weili/haiyang/outputs/infp_audio_5k_8_56_20250206-0737/test_0/audio_only/single_reconstruct"
    pred_videos = [os.path.join(video_pred_path, v) for v in os.listdir(video_pred_path) if v.endswith(".mp4")]
    gt_videos = [os.path.join(gt_path, v) for v in os.listdir(gt_path) if v.endswith(".mp4")]
    avg_score = eval_syncnet_videos(pred_videos, "./syncnet_eval.json", num_workers=8)
    print("avg syncnet conf:", avg_score)
