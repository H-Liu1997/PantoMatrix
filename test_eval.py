import sys
import os
metric_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "metric"))
sys.path.append(metric_path)
import torch.multiprocessing as mp
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow INFO and WARNING logs

def evaluate_all_metrics(video_pred_path, gt_path, verbose=False, csim=False):
    mp.set_start_method("spawn", force=True)
    from metric.syncnet_api import eval_syncnet_videos
    from metric.diversity_api import eval_diversity_videos, calcuate_sid
    from metric.ssim_api import video_level_evaluation
    from metric.csim_api import eval_csim_videos
    
    pred_videos = [os.path.join(video_pred_path, v) for v in os.listdir(video_pred_path) if v.endswith(".mp4")]
    gt_videos = [os.path.join(gt_path, v) for v in os.listdir(gt_path) if v.endswith(".mp4")]
    json_root_path = os.path.dirname(video_pred_path)
    syncnet_json = os.path.join(json_root_path, "syncnet_result.json")
    diversity_pred_json = os.path.join(json_root_path, "diversity_result_pred.json")
    diversity_gt_json = os.path.join(json_root_path, "diversity_result_gt.json")
    csim_json = os.path.join(json_root_path, "csim_result.json")
    
    start = time.time()
    avg_conf = eval_syncnet_videos(pred_videos, syncnet_json, num_workers=8)
    if verbose: print("syncnet time: ", time.time()-start)
    start = time.time()
    head_var_pred, exp_var_pred, all_motion_pred = eval_diversity_videos(pred_videos, diversity_pred_json, num_workers=8)
    if verbose: print("diversity time: ", time.time()-start)
    
    start = time.time()
    head_var_gt, exp_var_gt, all_motion_gt = eval_diversity_videos(gt_videos, diversity_gt_json, num_workers=8)
    sid_head = calcuate_sid(all_motion_gt, all_motion_pred, type='head')
    sid_exp = calcuate_sid(all_motion_gt, all_motion_pred, type='exp')
    if verbose: print("sid time: ", time.time()-start)
    
    if csim:
        start = time.time()
        csim_score = eval_csim_videos(pred_videos, csim_json)
        if verbose: print("csim time: ", time.time()-start)
    else:
        csim_score = 0.0
    
    start = time.time()
    ssim_results = video_level_evaluation(pred_videos, gt_videos)
    if verbose: print("ssim time: ", time.time()-start)
    
    if verbose: print("avg_conf: ", avg_conf)
    if verbose: print("head_var_pred: ", head_var_pred)
    if verbose: print("exp_var_pred: ", exp_var_pred)
    # if verbose: print("head_var_gt: ", head_var_gt)
    # if verbose: print("exp_var_gt: ", exp_var_gt)
    if verbose: print("sid_head: ", sid_head)
    if verbose: print("sid_exp: ", sid_exp)
    if verbose: print("csim_score: ", csim_score)
    if verbose: print("ssim: ", ssim_results.get("ssim"))
    if verbose: print("psnr: ", ssim_results.get("psnr"))
    if verbose: print("fvd: ", ssim_results.get("fvd"))
    if verbose: print("lpips: ", ssim_results.get("lpips"))
    
    return {
        "avg_conf": avg_conf,
        "head_var_pred": head_var_pred,
        "exp_var_pred": exp_var_pred,
        # "head_var_gt": head_var_gt,
        # "exp_var_gt": exp_var_gt,
        "sid_head": sid_head,
        "sid_exp": sid_exp,
        "csim_score": csim_score,
        "ssim": ssim_results.get("ssim"),
        "psnr": ssim_results.get("psnr"),
        "fvd": ssim_results.get("fvd"),
        "lpips": ssim_results.get("lpips"),
    }
    
if __name__ == "__main__":
    gt_video_folder = "/home/weili/haiyang/PantoMatrix/HDTF/cache_merge_v4"
    video_folder = "/home/weili/haiyang/outputs/infp_audio_5k_8_56_20250206-0737/test_0/audio_only/single_reconstruct"
    _ = evaluate_all_metrics(video_folder, gt_video_folder, verbose=False, csim=False)

