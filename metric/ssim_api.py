import os
import io
import cv2
import time
import torch
import lpips
import numpy as np
from PIL import Image
from decord import VideoReader
from torchvision import transforms
from scipy import linalg
from skimage.metrics import structural_similarity
from pytorch_i3d import InceptionI3d
from tqdm import tqdm
import torch
from pytorch_msssim import ms_ssim
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path):
    try:
        with open(model_path, 'rb') as f:
            buffer = io.BytesIO(f.read())
        state_dict = torch.load(buffer, map_location=device)
        return state_dict
    except Exception as e:
        print(e)
        return None
        
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    diff = mu1 - mu2
    offset = np.eye(sigma1.shape[0]) * eps
    covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))

def frechet_distance(samples_A, samples_B):
    A_mu = np.mean(samples_A, axis=0)
    A_sigma = np.cov(samples_A, rowvar=False)
    B_mu = np.mean(samples_B, axis=0)
    B_sigma = np.cov(samples_B, rowvar=False)
    try:
        dist = calculate_frechet_distance(A_mu, A_sigma, B_mu, B_sigma)
    except ValueError:
        dist = 1e+10
    return dist

loss_fn_alex = lpips.LPIPS(net='alex').to("cuda")
model_path = '/home/weili/haiyang/PantoMatrix/metric/model_weight/rgb_charades.pt'
i3d = InceptionI3d(400, in_channels=3)
i3d.replace_logits(157)
state_dict = load_model(model_path)
i3d.load_state_dict(state_dict)
i3d.to(device).eval()

def get_fvd(video1, video2):
    video1 = video1.unsqueeze(0)
    video2 = video2.unsqueeze(0)
    with torch.no_grad():
        feat1 = i3d.extract_features(video1).cpu().numpy()
        feat2 = i3d.extract_features(video2).cpu().numpy()
    return feat1, feat2

def lpips_metric(vid1, vid2, batch_size=240):
    chunks, remain = divmod(len(vid1), batch_size)
    all_lpips, count = 0, 0
    with torch.no_grad():
        for i in range(chunks):
            p = vid1[i*batch_size:(i+1)*batch_size]
            g = vid2[i*batch_size:(i+1)*batch_size]
            d = loss_fn_alex(p, g).mean().item()
            all_lpips += d*batch_size
            count += batch_size
        if remain > 0:
            p = vid1[chunks*batch_size:]
            g = vid2[chunks*batch_size:]
            d = loss_fn_alex(p, g).mean().item()
            all_lpips += d*remain
            count += remain
    return all_lpips / count

def psnr(vid1, vid2):
    mse = np.mean((vid1 - vid2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def ssim_metric(vid1, vid2):
    vals = []
    for f1, f2 in zip(vid1, vid2):
        g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
        s = structural_similarity(f1, f2, data_range=255.0, win_size=7)
        vals.append(s)
    return np.mean(vals)

def ms_ssim_metric(vid1, vid2):
    vid1_np = np.array(vid1)  
    vid2_np = np.array(vid2)
    vid1_tensor = torch.from_numpy(vid1_np).permute(0, 3, 1, 2).float().to("cuda")/255.0
    vid2_tensor = torch.from_numpy(vid2_np).permute(0, 3, 1, 2).float().to("cuda")/255.0
    # print(vid1_tensor.min(), vid2_tensor.max())  
    return ms_ssim(vid1_tensor, vid2_tensor, data_range=1.0).item()


def video_level_evaluation(pred_path_list, gt_path_list):
    psnr_val, ssim_val, lpips_val = 0, 0, 0
    all_gt, all_pred = [], []
    for p_path, g_path in tqdm(zip(pred_path_list, gt_path_list)):
        p_reader = VideoReader(p_path)
        g_reader = VideoReader(g_path)
        p_frames = []
        g_frames = []
        for i in range(len(p_reader)):
            p_img = p_reader[i].asnumpy()
            g_img = g_reader[i].asnumpy()
            if p_img.shape[2] == 4:
                p_img = cv2.cvtColor(p_img, cv2.COLOR_RGBA2BGR)
            if g_img.shape[2] == 4:
                g_img = cv2.cvtColor(g_img, cv2.COLOR_RGBA2BGR)
            p_frames.append(p_img)
            g_frames.append(g_img)
        p_arr = np.array(p_frames)
        g_arr = np.array(g_frames)
        psnr_val += psnr(p_arr, g_arr)
        # print(psnr_val)
        ssim_val += ms_ssim_metric(p_frames, g_frames)
        # print(ssim_val)
        
        # tensor
        g_frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in g_frames]
        p_frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in p_frames]
        p_arr_rgb = np.array(p_frames_rgb)
        g_arr_rgb = np.array(g_frames_rgb)
        p_tensor = torch.from_numpy(p_arr_rgb).permute(0, 3, 1, 2).float().to("cuda")/255.0*2-1
        g_tensor = torch.from_numpy(g_arr_rgb).permute(0, 3, 1, 2).float().to("cuda")/255.0*2-1
        # resize to 224x224
        p_tensor_512 = torch.nn.functional.interpolate(p_tensor, size=(512, 512))
        g_tensor_512 = torch.nn.functional.interpolate(g_tensor, size=(512, 512))
        g_tensor = torch.nn.functional.interpolate(g_tensor, size=(224, 224))
        p_tensor = torch.nn.functional.interpolate(p_tensor, size=(224, 224))
        
        p_tensor_bgr = p_tensor_512[:, [2, 1, 0], :, :]
        g_tensor_bgr = g_tensor_512[:, [2, 1, 0], :, :]
        lpips_val += lpips_metric(p_tensor_bgr, g_tensor_bgr)
        # print(lpips_val)
        p_tensor = p_tensor.permute(1, 0, 2, 3)
        g_tensor = g_tensor.permute(1, 0, 2, 3)
        # print(p_tensor.shape, g_tensor.shape)
        gt_fea, pred_fea = get_fvd(p_tensor, g_tensor)
        # print(gt_fea.shape, pred_fea.shape)
        all_gt.append(gt_fea.squeeze(0).squeeze(2).squeeze(2))
        all_pred.append(pred_fea.squeeze(0).squeeze(2).squeeze(2))
    all_gt = np.concatenate(all_gt, axis=1)
    all_pred = np.concatenate(all_pred, axis=1)
    fvd_val = frechet_distance(all_gt, all_pred)
    n = len(pred_path_list)
    return {
        "fvd": fvd_val,
        "lpips": lpips_val / n,
        "psnr": psnr_val / n,
        "ssim": ssim_val / n
    }
    
if __name__ == "__main__":
    pred_root = "/home/weili/haiyang/outputs/infp_audio_5k_8_56_20250206-0737/test_0/audio_only/single_reconstruct"
    gt_root = "/home/weili/haiyang/PantoMatrix/HDTF/cache_merge_v4"
    pred_list = [os.path.join(pred_root, f) for f in os.listdir(pred_root) if f.endswith(".mp4")]
    gt_list = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith(".mp4")]
    results = video_level_evaluation(pred_list, gt_list)
    print(results)
