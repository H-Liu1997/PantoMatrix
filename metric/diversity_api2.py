import numpy as np
from scipy import linalg
from sklearn.cluster import KMeans
import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from decord import VideoReader
from datasets.face_detector import FaceDetector
import urllib.request
from tqdm import tqdm
# Suppress TensorFlow Lite and OpenGL logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 1 = INFO, 2 = INFO+WARNING, 3 = INFO+WARNING+ERROR
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'  # Disable GPU logs for MediaPipe (if applicable)
os.environ['EGL_LOG_LEVEL'] = 'fatal'  # Suppress EGL initialization logs
os.environ['QT_LOGGING_RULES'] = "*=false"  # Suppress OpenGL Qt logs (if using Qt)
os.environ['GLOG_minloglevel'] = '3'  # Suppress OpenGL logs used by TensorFlow and MediaPipe
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)  # Show only errors


def calculate_activation_statistics(activations):
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape and sigma1.shape == sigma2.shape

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        covmean = linalg.sqrtm((sigma1 + np.eye(sigma1.shape[0]) * eps).dot(sigma2 + np.eye(sigma2.shape[0]) * eps))

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean.real if np.iscomplexobj(covmean) else covmean)

def calculate_variance(activations):
    return np.sum(np.var(activations, axis=0))

def matrix_to_rodrigues(R):
    angle = np.arccos((np.trace(R)-1)/2)
    axis = np.array([R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]])
    return axis / np.linalg.norm(axis) * angle

def get_head_motion(matrix):
    t = matrix[:3, 3]
    R = matrix[:3, :3] / np.linalg.norm(matrix[:3, :3], axis=0)
    return t, matrix_to_rodrigues(R)

def process_video_3bbox(video_path, detector):
    vr = VideoReader(video_path)
    frames = vr.get_batch(range(len(vr))).asnumpy()
    frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
    transformed_frames = F.interpolate(frames_tensor, size=(512, 512), mode='bicubic', align_corners=False)
    frames = transformed_frames.permute(0, 2, 3, 1).numpy()

    out_list = []
    for f in frames.astype(np.uint8):
        res = detector.get_face_xy_rotation_and_keypoints(f, 1.4, 1.6)
        if len(res[8]) == 0:
            return None
        t, rod = get_head_motion(np.array(res[10][0]))
        out_list.append(np.concatenate([t, rod, np.array(res[9][0])]))
    return np.array(out_list)

def calcuate_sid(gt, pred, type='exp'):
    # gt: list of [seq_len, dim]
    # pred: list of [seq_len, dim]
    if type == 'exp':
        k = 40
    else:
        k = 20
    merge_gt = np.concatenate(gt, axis=0)
    if type == 'exp':
        merge_gt = merge_gt[:, 6:]
    else:
        merge_gt = merge_gt[:, :6]
    # run kmeans on gt
    kmeans_gt = KMeans(n_clusters=k, random_state=0, n_init='auto').fit(merge_gt)
    # run kmeans on pred
    merge_pred = np.concatenate(pred, axis=0)
    if type == 'exp':
        merge_pred = merge_pred[:, 6:]
    else:
        merge_pred = merge_pred[:, :6]
    kmeans_pred = kmeans_gt.predict(merge_pred)
    # compute histogram
    hist_cnt = [0] * k
    for i in range(len(kmeans_pred)):
        hist_cnt[kmeans_pred[i]] += 1
    hist_cnt = np.array(hist_cnt)
    hist_cnt = hist_cnt / np.sum(hist_cnt)
    # compute entropy
    entropy = 0
    eps = 1e-6
    for i in range(k):
        entropy += hist_cnt[i] * np.log2(hist_cnt[i]+eps)
    return -entropy

class VideoDataset(Dataset):
    def __init__(self, video_list):
        self.video_list = video_list

    def __len__(self):
        return len(self.video_list)
    
    def __getitem__(self, index):
        video_path = self.video_list[index]
        detector = FaceDetector(VideoDataset.face_landmarker_path, face_detection_confidence=0.5, num_faces=5)
        motion_seq = process_video_3bbox(video_path, detector)
        if motion_seq is None:
            return None
        return calculate_variance(motion_seq[:, :6]), calculate_variance(motion_seq[:, 6:]), len(motion_seq), video_path, motion_seq

face_landmarker_path = "/home/weili/haiyang/PantoMatrix/face_landmarker.task"
if not os.path.exists(face_landmarker_path):
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        face_landmarker_path
    )
VideoDataset.face_landmarker_path = face_landmarker_path   

def eval_diversity_videos(video_path_list, output_json, num_workers=8):
    dataset = VideoDataset(video_path_list)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=num_workers, collate_fn=lambda x: x[0], drop_last=False, shuffle=False)

    results, all_motion = [], []
    for data in tqdm(dataloader, total=len(video_path_list)):
        if data is None:
            continue
        hv, ev, ln, video_path, motion_seq = data
        results.append({"video": video_path, "head_vairance": hv, "expression_vairance": ev, "length": ln})
        all_motion.append(motion_seq)

    with open(output_json, "w") as f:
        json.dump(results, f)
        
    all_head_var, all_exp_val, all_len = 0, 0, 0
    for res in results:
        all_head_var += res["head_vairance"]*res["length"]
        all_exp_val += res["expression_vairance"]*res["length"]
        all_len += res["length"]
    all_head_var /= all_len
    all_exp_val /= all_len
    return all_head_var, all_exp_val, all_motion

if __name__ == "__main__":
    # Store the path as a class attribute to avoid re-downloading in each worker
    gt_path = "/home/weili/haiyang/PantoMatrix/HDTF/cache_merge_v4"
    video_pred_path = "/home/weili/haiyang/outputs/infp_audio_5k_8_56_20250206-0737/test_0/audio_only/single_reconstruct"
    pred_videos = [os.path.join(video_pred_path, v) for v in os.listdir(video_pred_path) if v.endswith(".mp4")]
    gt_videos = [os.path.join(gt_path, v) for v in os.listdir(gt_path) if v.endswith(".mp4")]
    all_head_var, all_exp_val, pred_motion = eval_diversity_videos(pred_videos, "./div_eval_pred.json", num_workers=8)
    print(all_head_var, all_exp_val)
    gt_head_var, gt_exp_val, gt_motion = eval_diversity_videos(gt_videos, "./div_eval_gt.json", num_workers=8)
    print(gt_head_var, gt_exp_val)
    print(calcuate_sid(gt_motion, pred_motion, type='exp'))
    print(calcuate_sid(gt_motion, pred_motion, type='head'))