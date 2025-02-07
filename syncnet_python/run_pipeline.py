#!/usr/bin/python
import time, argparse, subprocess, pickle, os, glob, cv2, numpy as np
from shutil import rmtree
from scipy.interpolate import interp1d
from scipy.io import wavfile
from scipy import signal
import python_speech_features
import torch
from detectors import S3FD
from SyncNetInstance import SyncNetInstance
import scenedetect
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector
import shutil

def bb_iou(a,b):
    x1,y1,x2,y2=a
    X1,Y1,X2,Y2=b
    iw=min(x2,X2)-max(x1,X1)
    ih=min(y2,Y2)-max(y1,Y1)
    if iw<0 or ih<0: return 0
    inter=iw*ih
    s1=(x2-x1)*(y2-y1)
    s2=(X2-X1)*(Y2-Y1)
    return inter/(s1+s2-inter+1e-8)

def track_shot(faces, min_track=100, min_face_size=100, num_failed_det=25):
    tracks=[]
    while True:
        track=[]
        for frm in faces:
            for face in frm:
                if not track:
                    track.append(face)
                    frm.remove(face)
                elif face['frame']-track[-1]['frame']<=num_failed_det:
                    if bb_iou(face['bbox'],track[-1]['bbox'])>0.5:
                        track.append(face)
                        frm.remove(face)
                        continue
                else:
                    break
        if not track: break
        if len(track)>min_track:
            fnum=np.array([f['frame'] for f in track])
            bbox=np.array([f['bbox'] for f in track])
            rng=np.arange(fnum[0],fnum[-1]+1)
            out=[]
            for i in range(4):
                out.append(np.interp(rng,fnum,bbox[:,i]))
            out=np.stack(out,1)
            if max(np.mean(out[:,2]-out[:,0]),np.mean(out[:,3]-out[:,1]))>min_face_size:
                tracks.append({'frame':rng,'bbox':out})
    return tracks

def crop_video(img_folder_path, track, savepath, frame_rate=25, crop_scale=0.4):
    flist=sorted(glob.glob(os.path.join(img_folder_path,'*.jpg')))
    vwriter = cv2.VideoWriter(savepath+'t.mp4', cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (224,224))
    d={'x':[],'y':[],'s':[]}
    for b in track['bbox']:
        d['s'].append(max(b[3]-b[1],b[2]-b[0])/2)
        d['y'].append((b[1]+b[3])/2)
        d['x'].append((b[0]+b[2])/2)
    d['s']=signal.medfilt(d['s'],13)
    d['x']=signal.medfilt(d['x'],13)
    d['y']=signal.medfilt(d['y'],13)
    for i,f in enumerate(track['frame']):
        bs=d['s'][i]
        img=cv2.imread(flist[f])
        pad=int(bs*(1+2*crop_scale))
        tmp=np.pad(img,((pad,pad),(pad,pad),(0,0)),'constant',constant_values=110)
        my=d['y'][i]+pad
        mx=d['x'][i]+pad
        face=tmp[int(my-bs):int(my+bs*(1+2*crop_scale)),
                 int(mx-bs*(1+crop_scale)):int(mx+bs*(1+crop_scale))]
        vwriter.write(cv2.resize(face,(224,224)))
        # print(i)
    vwriter.release()
    return savepath+'t.mp4'

def bbox_detection(img_folder_path, det, facedet_scale=0.25):
    flist=sorted(glob.glob(os.path.join(img_folder_path,'*.jpg')))
    out=[]
    for i,fname in enumerate(flist):
        img=cv2.imread(fname)
        bboxes=det.detect_faces(cv2.cvtColor(img,cv2.COLOR_BGR2RGB),0.9,[facedet_scale])
        out.append([])
        for b in bboxes:
            out[-1].append({'frame':i,'bbox':b[:-1].tolist(),'conf':b[-1]})
        # print(i)
    return out
  
def scene_detect(video_path):
    vm=VideoManager([video_path])
    sm=StatsManager()
    sc=SceneManager(sm)
    sc.add_detector(ContentDetector())
    vm.set_downscale_factor()
    vm.start()
    sc.detect_scenes(frame_source=vm)
    slist=sc.get_scene_list(vm.get_base_timecode())
    if not slist:
        slist=[(vm.get_base_timecode(),vm.get_current_timecode())]
    return slist

import moviepy.editor as mp
def evaluate(video_path, audio_path, allconf, det, s, img_folder_1, img_folder_2):
    os.makedirs(img_folder_1, exist_ok=True)
    os.makedirs(img_folder_2, exist_ok=True)
    
    # Combine video and audio
    output_video_audio_path = img_folder_1 + "video_audio.mp4"
    vclip = mp.VideoFileClip(video_path)
    aclip = mp.AudioFileClip(audio_path)
    final_clip = vclip.set_audio(aclip)
    final_clip.write_videofile(output_video_audio_path, codec="libx264", audio_codec="aac")
    vclip.close()
    aclip.close()
            
    # resample to 25fps video
    output_video_path = img_folder_1 + "video.mp4"
    subprocess.call(
        f'ffmpeg -y -i "{output_video_audio_path}" -vf "fps=25" "{output_video_path}"',
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )  
    subprocess.call(
        f'ffmpeg -y -i "{output_video_path}" -threads 1 -f image2 {img_folder_1}/%06d.jpg',
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    faces=bbox_detection(img_folder_1, det)
    scene=scene_detect(output_video_path)
    alltracks=[]
    # print(scene, output_video_path)
    for shot in scene:
        start=shot[0].frame_num
        end=shot[1].frame_num
        if end-start>=100:
            alltracks.extend(track_shot(faces[start:end]))
    vidtracks=[]
    for i,t in enumerate(alltracks):
        vidtracks.append(crop_video(img_folder_1, t, img_folder_2, frame_rate=25))
    
    output_video_audio_path = img_folder_2 + 't2.mp4'
    vclip = mp.VideoFileClip(img_folder_2 + 't.mp4')
    aclip = mp.AudioFileClip(audio_path)
    final_clip = vclip.set_audio(aclip)
    final_clip.write_videofile(output_video_audio_path, codec="libx264", audio_codec="aac")
    vclip.close()
    aclip.close()
    
    allconf=[]
    for path in vidtracks:
        conf=s.evaluate(output_video_audio_path)
        allconf.extend(conf)
        
    shutil.rmtree(img_folder_1)
    shutil.rmtree(img_folder_2)
    return allconf

# single video test
# video_path = "/home/weili/haiyang/PantoMatrix/HDTF/cache_ori_v4/RD_Radio10_000_000.mp4"
# audio_path = "/home/weili/haiyang/PantoMatrix/HDTF/cache_audio_v4/RD_Radio10_000_000.wav"
# img_folder_1 = "./data/tmp1/" 
# img_folder_2 = "./data/tmp2/"

# det=S3FD(device='cuda')
# s=SyncNetInstance(vshift=15, batch_size=20)
# s.loadParameters("data/syncnet_v2.model")

# all_conf = []
# all_conf = evaluate(video_path, audio_path, all_conf, det, s, img_folder_1, img_folder_2)
# print(sum(all_conf)/len(all_conf))

from tqdm import tqdm
import json
video_folder = "/home/weili/haiyang/PantoMatrix/HDTF/cache_ori_v4/"
audio_folder = "/home/weili/haiyang/PantoMatrix/HDTF/cache_audio_v4/"
img_folder_1 = "./data/tmp1/"
img_folder_2 = "./data/tmp2/"
det=S3FD(device='cuda')
s=SyncNetInstance(vshift=15, batch_size=20)
s.loadParameters("data/syncnet_v2.model")
test_path = ["/home/weili/haiyang/PantoMatrix/datasets/data_json/infp_s20_l64_kw2_na2.json"]
test_list = []
for data_meta_path in test_path:
    test_list.extend(json.load(open(data_meta_path, "r")))
test_list = [item for item in test_list if item.get("mode") == "test_wild"]
seen_ids = set()
test_list = [item for item in test_list if not (item["video_id"] in seen_ids or seen_ids.add(item["video_id"]))]

import shutil
import zipfile
import os

all_conf = []
# zip_filename = "output_videos_audios.zip"

# with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
for video_json in tqdm(test_list):
    video_id = video_json["video_id"]
    video_path = video_folder + video_id + ".mp4"
    audio_path = audio_folder + video_id + ".wav"
    print(video_path)
    # Process the video and audio
    all_conf = evaluate(video_path, audio_path, all_conf, det, s, img_folder_1, img_folder_2)
    print(sum(all_conf) / len(all_conf))
        # Add video and audio to the zip file
        # if os.path.exists(video_path):
        #     zipf.write(video_path, arcname=os.path.basename(video_path))
        # if os.path.exists(audio_path):
        #     zipf.write(audio_path, arcname=os.path.basename(audio_path))

print(sum(all_conf) / len(all_conf))  # Print the final confidence score
# print(f"Saved videos and audios in {zip_filename}")
