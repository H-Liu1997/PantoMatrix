# level_1_code.py

import torch.multiprocessing as mp
import os

def main():
    # IMPORTANT: set start method FIRST, before any GPU usage
    mp.set_start_method("spawn", force=True)

    # Now import or call any modules that might use the GPU
    from syncnet_api import eval_syncnet_videos
    from diversity_api import eval_diversity_videos, calcuate_sid
    from ssim_api import video_level_evaluation
    from csim_api import eval_csim_videos

    # Then do the rest of your main logic
    video_folder = "/home/weili/haiyang/outputs/infp_audio_5k_8_56_20250206-0737/test_0/audio_only/single_reconstruct"
    video_name = os.listdir(video_folder)
    video_path_list = [os.path.join(video_folder, vpath) for vpath in video_name if vpath.endswith(".mp4")]

    # Example usage
    avg_head_vairance, avg_expression_vairance, all_motion_pred = eval_diversity_videos(video_path_list, "./diversity_result_pred.json", num_workers=8)
    print("avg_head_vairance:", avg_head_vairance)
    print("avg_expression_vairance:", avg_expression_vairance)

    # etc. for syncnet, csim, etc...

if __name__ == "__main__":
    main()
