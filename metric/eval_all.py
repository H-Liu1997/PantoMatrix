import os
import torch.multiprocessing as mp
import time
mp.set_start_method("spawn", force=True)
import argparse

def evaluate_all_metrics(video_pred_path, gt_path, verbose=False, csim=True):
    from metric.syncnet_api import eval_syncnet_videos
    from metric.diversity_api import eval_diversity_videos, calcuate_sid
    from metric.ssim_api import video_level_evaluation
    from metric.csim_api import eval_csim_videos
    
    pred_videos = [os.path.join(video_pred_path, v) for v in os.listdir(video_pred_path) if v.endswith(".mp4")]
    gt_videos = [os.path.join(gt_path, v) for v in os.listdir(gt_path) if v.endswith(".mp4")]
    json_root_path = video_pred_path
    syncnet_json = os.path.join(json_root_path, "syncnet_result.json")
    diversity_pred_json = os.path.join(json_root_path, "diversity_result_pred.json")
    diversity_gt_json = os.path.join(json_root_path, "diversity_result_gt.json")
    csim_json = os.path.join(json_root_path, "csim_result.json")
    
    start = time.time()
    avg_conf = eval_syncnet_videos(pred_videos, syncnet_json, num_workers=8)
    if verbose: print("syncnet time: ", time.time()-start)
    # start = time.time()
    # head_var_pred, exp_var_pred, all_motion_pred = eval_diversity_videos(pred_videos, diversity_pred_json, num_workers=8)
    # if verbose: print("diversity time: ", time.time()-start)
    
    # start = time.time()
    # head_var_gt, exp_var_gt, all_motion_gt = eval_diversity_videos(gt_videos, diversity_gt_json, num_workers=8)
    # sid_head = calcuate_sid(all_motion_gt, all_motion_pred, type='head')
    # sid_exp = calcuate_sid(all_motion_gt, all_motion_pred, type='exp')
    # if verbose: print("sid time: ", time.time()-start)
    
    # if csim:
    #     start = time.time()
    #     csim_score = eval_csim_videos(pred_videos, csim_json)
    #     if verbose: print("csim time: ", time.time()-start)
    # else:
    #     csim_score = 0.0
    
    # start = time.time()
    # ssim_results = video_level_evaluation(pred_videos, gt_videos)
    # if verbose: print("ssim time: ", time.time()-start)

    if verbose: print("avg_conf: ", avg_conf)
    # if verbose: print("head_var_pred: ", head_var_pred)
    # if verbose: print("exp_var_pred: ", exp_var_pred)
    # # if verbose: print("head_var_gt: ", head_var_gt)
    # # if verbose: print("exp_var_gt: ", exp_var_gt)
    # if verbose: print("sid_head: ", sid_head)
    # if verbose: print("sid_exp: ", sid_exp)
    # if verbose: print("csim_score: ", csim_score)
    # if verbose: print("ssim: ", ssim_results.get("ssim"))
    # if verbose: print("psnr: ", ssim_results.get("psnr"))
    # if verbose: print("fvd: ", ssim_results.get("fvd"))
    # if verbose: print("lpips: ", ssim_results.get("lpips"))
    
    save_text_path = os.path.join(video_pred_path, "metrics.txt")
    # print(save_text_path)
    # avg_conf = 0.0
    # head_var_pred = 0.0
    # exp_var_pred = 0.0
    # # head_var_gt = 0.0
    # # exp_var_gt = 0.0
    # sid_head = 0.0
    # sid_exp = 0.0
    # csim_score = 0.0
    # ssim_results = {
    #     "ssim": 0.0,
    #     "psnr": 0.0,
    #     "fvd": 0.0,
    #     "lpips": 0.0
    # }
    
    with open(save_text_path, "w") as f:
        f.write("avg_conf: {}\n".format(avg_conf))
        # f.write("head_var_pred: {}\n".format(head_var_pred))
        # f.write("exp_var_pred: {}\n".format(exp_var_pred))
        # f.write("sid_head: {}\n".format(sid_head))
        # f.write("sid_exp: {}\n".format(sid_exp))
        # f.write("csim_score: {}\n".format(csim_score))
        # f.write("ssim: {}\n".format(ssim_results.get("ssim")))
        # f.write("psnr: {}\n".format(ssim_results.get("psnr")))
        # f.write("fvd: {}\n".format(ssim_results.get("fvd")))
        # f.write("lpips: {}\n".format(ssim_results.get("lpips")))
        
    return {
        "avg_conf": avg_conf,
        # "head_var_pred": head_var_pred,
        # "exp_var_pred": exp_var_pred,
        # # "head_var_gt": head_var_gt,
        # # "exp_var_gt": exp_var_gt,
        # "sid_head": sid_head,
        # "sid_exp": sid_exp,
        # "csim_score": csim_score,
        # "ssim": ssim_results.get("ssim"),
        # "psnr": ssim_results.get("psnr"),
        # "fvd": ssim_results.get("fvd"),
        # "lpips": ssim_results.get("lpips"),
    }
    
if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--video_pred_path", type=str, default="/home/weili/haiyang/outputs/infp_audio_5k_8_56_20250206-0737/test_0/audio_only/single_reconstruct")
    arg_parser.add_argument("--gt_path", type=str, default="/home/weili/haiyang/PantoMatrix/HDTF/cache_merge_v4")
    args = arg_parser.parse_args()
    _ = evaluate_all_metrics(args.video_pred_path, args.gt_path, verbose=True, csim=False)

