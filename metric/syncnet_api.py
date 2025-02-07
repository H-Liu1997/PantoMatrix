import os
import json
import torch.multiprocessing as mp
from tqdm import tqdm
from detectors import S3FD
from SyncNetInstance import SyncNetInstance
from run_pipeline import syncnet_pipeline
from run_syncnet import syncnet_run
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

_detector = None
_syncnet = None

def worker_init():
    mp.set_start_method('spawn', force=True)
    global _detector, _syncnet
    _detector = S3FD(device="cuda")
    _syncnet = SyncNetInstance()
    _syncnet.loadParameters("/home/weili/haiyang/PantoMatrix/metric/model_weight/syncnet_v2.model")

def process_video(video_path):
    vid_name = os.path.basename(video_path)
    dir_name = os.path.dirname(video_path)
    tmp_dir = os.path.join(dir_name, vid_name[:-4])
    # print(f"Processing {vid_name} in {tmp_dir}")
    os.makedirs(tmp_dir, exist_ok=True)
    syncnet_pipeline(_detector, video_path, vid_name, data_dir=tmp_dir)
    _, conf, _, _, frames = syncnet_run(_syncnet, video_path, vid_name, data_dir=tmp_dir)
    os.system(f"rm -rf {tmp_dir}")
    return conf, len(frames), vid_name

def eval_syncnet_videos(video_path_list, output_json, num_workers=8):
    results = []
    with mp.Pool(num_workers, initializer=worker_init) as pool:
        for c, ln, vid in tqdm(pool.imap(process_video, video_path_list), total=len(video_path_list)):
            results.append({"video": vid, "conf": c.item(), "length": ln})
        pool.close()
        pool.join()
    total_conf, total_len = 0, 0
    for r in results:
        total_conf += r["conf"] * r["length"]
        total_len += r["length"]
    avg_conf = total_conf / total_len if total_len else 0
    with open(output_json, "w") as f:
        json.dump(results, f)
    return avg_conf

if __name__ == "__main__":
    vs = ["/home/weili/haiyang/metric/syncnet_work/RD_Radio54_000_000/RD_Radio54_000_000.mp4"]
    avg_score = eval_syncnet_videos(vs, "./syncnet_eval.json", num_workers=4)
    print("avg syncnet conf:", avg_score)