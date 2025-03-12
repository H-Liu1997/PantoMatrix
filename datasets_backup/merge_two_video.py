import os
import numpy as np
from decord import VideoReader, cpu
import imageio
from moviepy.editor import VideoFileClip

def merge_videos(input_dir, video_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for file in os.listdir(input_dir):
        if file.endswith("_0.mp4"):
            base_name = file[:-6]
            video_0 = os.path.join(input_dir, file)
            video_1 = os.path.join(input_dir, f"{base_name}_1.mp4")
            gt_0 = os.path.join(video_dir, f"{base_name}_0.mp4")
            gt_1 = os.path.join(video_dir, f"{base_name}_1.mp4")
            
            print("Processing:", video_0, video_1, gt_0, gt_1)
            if not all(os.path.exists(v) for v in [video_0, video_1, gt_0, gt_1]):
                print(f"Skipping {base_name}, missing files.")
                continue
            
            merged_video_path = os.path.join(save_dir, f"{base_name}_merged.mp4")
            merge_grid(video_0, video_1, gt_0, gt_1, merged_video_path)
            print("Merged video saved:", merged_video_path)

def merge_grid(video_0, video_1, gt_0, gt_1, output_path):
    vr0 = VideoReader(video_0, ctx=cpu(0))
    vr1 = VideoReader(video_1, ctx=cpu(0))
    vrgt0 = VideoReader(gt_0, ctx=cpu(0))
    vrgt1 = VideoReader(gt_1, ctx=cpu(0))
    
    num_frames = min(len(vr0), len(vr1), len(vrgt0), len(vrgt1))
    frame_example = vr0[0].asnumpy()
    h, w, _ = frame_example.shape
    
    writer = imageio.get_writer(output_path, fps=vr0.get_avg_fps())
    for i in range(num_frames):
        frame0 = vr0[i].asnumpy()
        frame1 = vr1[i].asnumpy()
        frame_gt0 = vrgt0[i].asnumpy()
        frame_gt1 = vrgt1[i].asnumpy()
        
        top_row = np.hstack([frame0, frame1])
        bottom_row = np.hstack([frame_gt0, frame_gt1])
        merged_frame = np.vstack([top_row, bottom_row])
        writer.append_data(merged_frame)
    writer.close()
    
    merge_audio(video_0, output_path)

def merge_audio(source_video, target_video):
    video_clip = VideoFileClip(target_video)
    audio_clip = VideoFileClip(source_video).audio
    final_clip = video_clip.set_audio(audio_clip)
    temp_path = target_video.replace(".mp4", "_audio.mp4")
    final_clip.write_videofile(temp_path, codec="libx264", audio_codec="aac")
    final_clip.close()
    video_clip.close()
    audio_clip.close()
    os.replace(temp_path, target_video)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    args = parser.parse_args()
    
    merge_videos(args.input_dir, args.video_dir, args.save_dir)
