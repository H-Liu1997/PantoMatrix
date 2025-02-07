import os
import json
import torch
import numpy as np
from decord import VideoReader
from facenet_pytorch import InceptionResnetV1
from tqdm import tqdm

def process_video(video_path, batch_size=240, resnet=None):
    vd = VideoReader(video_path)
    f_np = vd.get_batch(range(len(vd))).asnumpy()
    f_ts = torch.from_numpy(f_np).permute(0, 3, 1, 2).float()
    f_rs = torch.nn.functional.interpolate(f_ts, (160, 160), mode='bilinear', align_corners=False)
    f_rs = f_rs / 255.0 * 2 - 1
    c, r = divmod(len(f_rs), batch_size)
    feats = []
    with torch.no_grad():
        for i in range(c):
            e = resnet(f_rs[i*batch_size:(i+1)*batch_size]).cpu()
            feats.append(e)
        if r:
            e = resnet(f_rs[-r:]).cpu()
            feats.append(e)
    emb = torch.cat(feats)
    ref = emb[0:1].repeat(emb.size(0), 1)
    cs = torch.nn.functional.cosine_similarity(emb, ref).numpy()[1:]
    return float(np.mean(cs)), len(vd)-1

def eval_csim_videos(video_paths, output_json):
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    rs = []
    all_csim, all_len = 0, 0
    for vp in tqdm(video_paths):
        csim, ln = process_video(vp, resnet=resnet)
        rs.append({"video": vp, "csim": csim, "length": ln})
    for r in rs:
        all_csim += r["csim"] * r["length"]
        all_len += r["length"]
    avg_csim = all_csim / all_len if all_len else 0
    with open(output_json, "w") as f:
        json.dump(rs, f)
    return avg_csim

if __name__ == "__main__":
    pred_root = "/home/weili/haiyang/PantoMatrix/HDTF/cache_merge_v4"
    pred_list = [os.path.join(pred_root, f) for f in os.listdir(pred_root) if f.endswith(".mp4")]
    avg_score = eval_csim_videos(pred_list, "./csim_eval.json")
    print("avg csim:", avg_score)
