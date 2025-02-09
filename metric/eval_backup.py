# from syncnet_api import eval_syncnet_videos
# from diversity_api import eval_diversity_videos, calcuate_sid
# from csim_api import eval_csim_videos
import json
import os

# video_folder = "/home/weili/haiyang/outputs/infp_audio_5k_8_56_20250206-0737/test_0/audio_only/single_reconstruct"
# video_name = os.listdir(video_folder)
# video_path_list = [os.path.join(video_folder, vpath) for vpath in video_name if vpath.endswith(".mp4")]
# print(video_path_list)

def parse_name(fname):
    base = fname[:-4]
    parts = base.split('_')
    return '_'.join(parts[:2]), '_'.join(parts[2:])

# merge audio and video
test_path = ["/home/weili/haiyang/PantoMatrix/datasets/data_json/infp_s20_l64_kw2_na2_v6.json"]
test_list = []
for data_meta_path in test_path:
    test_list.extend(json.load(open(data_meta_path, "r")))
test_list = [item for item in test_list if item.get("mode") == "test_wild"]
seen_ids = set()
test_list = [item for item in test_list if not (item["video_id"] in seen_ids or seen_ids.add(item["video_id"]))]
seen_ids = set()
test_list = [item for item in test_list if not (parse_name(item["video_id"])[0] in seen_ids or seen_ids.add(parse_name(item["video_id"])[0]))]
video_folder = "/home/weili/haiyang/PantoMatrix/HDTF/cache_ori_v6/"
merge_audio = True
if merge_audio:
    import os
    audio_folder = "/home/weili/haiyang/PantoMatrix/HDTF/cache_audio_v6/"
    merger_folder = "/home/weili/haiyang/PantoMatrix/HDTF/cache_merge_v6/"
    os.makedirs(merger_folder, exist_ok=True)
    for video_json in test_list:
        video_id = video_json["video_id"]
        video_path = video_folder + video_id + ".mp4"
        audio_path = audio_folder + video_id + ".wav"
        merger_path = merger_folder + video_id + ".mp4"
        os.system("ffmpeg -i {} -i {} -c:v copy -c:a aac -strict experimental {}".format(video_path, audio_path, merger_path))
    video_path_list = [merger_folder + video_json["video_id"] + ".mp4" for video_json in test_list]
else:
    video_path_list = [video_folder + video_json["video_id"] + ".mp4" for video_json in test_list]

# variance 
# avg_head_vairance, avg_expression_vairance, all_motion_pred = eval_diversity_videos(video_path_list, "./diversity_result_pred.json", num_workers=8)
# print("avg_head_vairance: ", avg_head_vairance)
# print("avg_expression_vairance: ", avg_expression_vairance)

# # sid
# gt_video_folder = "/home/weili/haiyang/PantoMatrix/HDTF/cache_merge_v4"
# gt_video_name = os.listdir(gt_video_folder)
# gt_video_path_list = [os.path.join(gt_video_folder, vpath) for vpath in gt_video_name if vpath.endswith(".mp4")]
# print(gt_video_path_list)
# avg_head_vairance, avg_expression_vairance, all_motion_gt= eval_diversity_videos(gt_video_path_list, "./diversity_result_gt.json", num_workers=8)
# sid_head = calcuate_sid(all_motion_gt, all_motion_pred, type='head')
# sid_exp = calcuate_sid(all_motion_gt, all_motion_pred, type='exp')
# print(sid_head, sid_exp)
# sid_head = calcuate_sid(all_motion_gt, all_motion_gt, type='head')
# sid_exp = calcuate_sid(all_motion_gt, all_motion_gt, type='exp')
# print(sid_head, sid_exp)

# syncnet
# avg_conf = eval_syncnet_videos(video_path_list, "./syncnet_result.json", num_workers=8)
# print("avg_conf: ", avg_conf)

# from syncnet_api import eval_syncnet_videos
# from diversity_api import eval_diversity_videos, calcuate_sid
# from ssim_api import video_level_evaluation
# from csim_api import eval_csim_videos
# import json
# import os

# video_folder = "/home/weili/haiyang/outputs/infp_audio_5k_8_56_20250206-0737/test_0/audio_only/single_reconstruct"
# video_name = os.listdir(video_folder)
# video_path_list = [os.path.join(video_folder, vpath) for vpath in video_name if vpath.endswith(".mp4")]
# #print(video_path_list)

# # ================== single video ================== 
# # syncnet
# # avg_conf = eval_syncnet_videos(video_path_list, "./syncnet_result.json", num_workers=8)
# # print("avg_conf: ", avg_conf)

# # variance 
# avg_head_vairance, avg_expression_vairance, all_motion_pred = eval_diversity_videos(video_path_list, "./diversity_result_pred.json", num_workers=8)
# print("avg_head_vairance: ", avg_head_vairance)
# print("avg_expression_vairance: ", avg_expression_vairance)

# # csim
# avg_score = eval_csim_videos(video_path_list, "./csim_result.json")
# print("avg_score: ", avg_score)

# ================== with gt ==================
# sid
# gt_video_folder = "/home/weili/haiyang/PantoMatrix/HDTF/cache_merge_v4"
# gt_video_name = os.listdir(gt_video_folder)
# gt_video_path_list = [os.path.join(gt_video_folder, vpath) for vpath in gt_video_name if vpath.endswith(".mp4")]
# # print(gt_video_path_list)

# avg_head_vairance, avg_expression_vairance, all_motion_gt= eval_diversity_videos(gt_video_path_list, "./diversity_result_gt.json", num_workers=8)
# sid_head = calcuate_sid(all_motion_gt, all_motion_pred, type='head')
# sid_exp = calcuate_sid(all_motion_gt, all_motion_pred, type='exp')
# print(sid_head, sid_exp)
# sid_head = calcuate_sid(all_motion_gt, all_motion_gt, type='head')
# sid_exp = calcuate_sid(all_motion_gt, all_motion_gt, type='exp')
# print(sid_head, sid_exp)

# # ssim, psnr, fvd, lpips
# results = video_level_evaluation(video_path_list, gt_video_path_list)
# print(results)
