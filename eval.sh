#!/usr/bin/env bash
python ./metric/eval_all.py --video_pred_path /home/weili/haiyang/outputs/infp_audio_5k_8_56_3k_20250209-0902/test_0/audio_only/single_reconstruct/
python ./metric/eval_all.py --video_pred_path /home/weili/haiyang/outputs/infp_audio_5k_8_56_3k_20250209-0902/test_160000/audio_only/single_reconstruct/
python ./metric/eval_all.py --video_pred_path /home/weili/haiyang/outputs/infp_audio_5k_8_56_3k_20250209-0902/test_165000/audio_only/single_reconstruct/

python ./metric/eval_all.py --video_pred_path /home/weili/haiyang/outputs/infp_audio_5k_8_56_300k_20250209-0905/test_0/audio_only/single_reconstruct/
python ./metric/eval_all.py --video_pred_path /home/weili/haiyang/outputs/infp_audio_5k_8_56_300k_20250209-0905/test_160000/audio_only/single_reconstruct/
python ./metric/eval_all.py --video_pred_path /home/weili/haiyang/outputs/infp_audio_5k_8_56_300k_20250209-0905/test_165000/audio_only/single_reconstruct/

python ./metric/eval_all.py --video_pred_path /home/weili/haiyang/PantoMatrix/HDTF/cache_merge_v6
python ./metric/eval_all.py --video_pred_path /home/weili/haiyang/PantoMatrix/HDTF/test_reconstructions_v4
python ./metric/eval_all.py --video_pred_path /home/weili/haiyang/PantoMatrix/HDTF/test_reconstructions_v6