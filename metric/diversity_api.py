"""
https://github.com/Boese0601/Dyadic-Interaction-Modeling/blob/main/code/metrics/eval_utils.py
"""
import numpy as np
from scipy import linalg
from sklearn.cluster import KMeans
from scipy.stats import entropy
import os
import json
import torch.multiprocessing as mp
from tqdm import tqdm
import torch
import torch.nn.functional as F
from decord import VideoReader
from tqdm import tqdm
from datasets.face_detector import FaceDetector
import urllib.request

def calculate_activation_statistics(activations):
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)

def calculate_variance(activations):
    return np.sum(np.var(activations, axis=0))

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
        
def sts(x, y, timestep=0.1):
    ans = 0
    total_sample, dim = x.shape
    for di  in range(dim):
        for i in range(1, total_sample):
            ans += ((x[i][di] - x[i-1][di]) - (y[i][di] - y[i-1][di]))**2 / timestep
    return np.sqrt(ans)

def worker_init():
    # mp.set_start_method('spawn', force=True)
    global detector
    face_landmarker_path = "/home/weili/haiyang/PantoMatrix/face_landmarker.task"
    if not os.path.exists(face_landmarker_path):
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, face_landmarker_path)
    detector = FaceDetector(face_landmarker_path, face_detection_confidence=0.5, num_faces=5)

def matrix_to_rodrigues(R):
    angle = np.arccos((np.trace(R)-1)/2)
    rx = R[2,1] - R[1,2]
    ry = R[0,2] - R[2,0]
    rz = R[1,0] - R[0,1]
    axis = np.array([rx, ry, rz])
    axis /= np.linalg.norm(axis)
    rod = axis * angle
    return rod

def get_head_motion(matrix):
    t = matrix[:3, 3]
    M = matrix[:3, :3]
    sx = np.linalg.norm(M[:, 0])
    sy = np.linalg.norm(M[:, 1])
    sz = np.linalg.norm(M[:, 2])
    R = M / [sx, sy, sz]
    rod = matrix_to_rodrigues(R)
    return t, rod

def process_video_3bbox(video_path, mouth_bbox_scale=1.4, eye_bbox_scale=1.6, detector=None):
    vr = VideoReader(video_path)
    frames = vr.get_batch(range(len(vr)))
    frames_tensor = torch.from_numpy(frames.asnumpy()).to('cuda').permute(0, 3, 1, 2).float()
    transformed_frames = F.interpolate(frames_tensor, size=(512, 512), mode='bicubic', align_corners=False)
    frames = transformed_frames.permute(0, 2, 3, 1).cpu().numpy()
    out_list = []
    for i in range(len(frames)):
        f = frames[i].astype(np.uint8)
        res = detector.get_face_xy_rotation_and_keypoints(f, mouth_bbox_scale, eye_bbox_scale)
        blendshapes = np.array(res[9][0])
        transformation_matrices = np.array(res[10][0])
        t, rod = get_head_motion(transformation_matrices)
        head_motion = np.concatenate([t, rod])
        if len(res[8]) == 0:
            return None
        out_list.append(np.concatenate([head_motion, blendshapes]))      
    return np.array(out_list)

def process_video(video_path):
    motion_seq = process_video_3bbox(video_path, detector=detector) # [seq_len, 6+52]
    head_vairance = calculate_variance(motion_seq[:, :6])
    expression_vairance = calculate_variance(motion_seq[:, 6:])
    # head_sid = calcuate_sid([motion_seq], [motion_seq], type='head')
    # exp_sid = calcuate_sid([motion_seq], [motion_seq], type='exp')
    return head_vairance, expression_vairance, len(motion_seq), video_path, motion_seq
   
def eval_diversity_videos(video_path_list, output_json, num_workers=8):
    results = []
    all_motion = []
    with mp.Pool(num_workers, initializer=worker_init) as pool:
        for hv, ev, ln, video_path, motion_seq in tqdm(pool.imap(process_video, video_path_list), total=len(video_path_list)):
            results.append({"video": video_path, "head_vairance": hv, "expression_vairance": ev, "length": ln})
            all_motion.append(motion_seq)
        pool.close()
        pool.join()
    avg_head_vairance, avg_expression_vairance, all_length = 0, 0, 0
    for res in results:
        avg_head_vairance += res["head_vairance"] * res["length"]
        avg_expression_vairance += res["expression_vairance"] * res["length"]
        all_length += res["length"]
    avg_head_vairance /= all_length
    avg_expression_vairance /= all_length
    with open(output_json, "w") as f:
        json.dump(results, f)
    return avg_head_vairance, avg_expression_vairance, all_motion   


if __name__ == "__main__":
    # 
    # pred_root = "/home/weili/haiyang/outputs/infp_audio_5k_8_56_20250206-0737/test_0/audio_only/single_reconstruct"
    # gt_root = "/home/weili/haiyang/PantoMatrix/HDTF/cache_merge_v4"
    
    vs = ["/home/weili/haiyang/outputs/infp_audio_5k_8_56_20250206-0737/test_0/audio_only/single_reconstruct/RD_Radio10_000_000.mp4"]
    output_json = "./div_eval_pred.json"
    avg_head_vairance, avg_expression_vairance, all_motion_pred = eval_diversity_videos(vs, output_json, 4)
    print(avg_head_vairance, avg_expression_vairance)
    
    vsgt = ["/home/weili/haiyang/PantoMatrix/HDTF/cache_merge_v4/RD_Radio10_000_000.mp4"]
    output_json_gt = "./div_eval_gt.json"
    avg_head_vairance, avg_expression_vairance, all_motion_gt = eval_diversity_videos(vsgt, output_json_gt, 4)
    print(avg_head_vairance, avg_expression_vairance)
    
    sid_head = calcuate_sid(all_motion_gt, all_motion_pred, type='head')
    sid_exp = calcuate_sid(all_motion_gt, all_motion_pred, type='exp')
    print(sid_head, sid_exp)