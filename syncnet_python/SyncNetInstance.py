import os
import cv2
import glob
import torch
import numpy as np
import subprocess
import tempfile
from scipy import signal
from scipy.io import wavfile
import librosa
import python_speech_features
import shutil

# import torch
# import numpy
# import time, pdb, argparse, subprocess, os, math, glob
# import cv2
# import python_speech_features
# from scipy import signal
# from scipy.io import wavfile
from SyncNetModel import *
from shutil import rmtree

def calc_pdist(feat1, feat2, vshift=10):
    win_size = vshift*2+1
    feat2p = torch.nn.functional.pad(feat2,(0,0,vshift,vshift))
    dists = []
    for i in range(0,len(feat1)):
        dists.append(torch.nn.functional.pairwise_distance(feat1[[i],:].repeat(win_size, 1), feat2p[i:i+win_size,:]))
    return dists

class SyncNetInstance(torch.nn.Module):
    def __init__(self, dropout=0, num_layers_in_fc_layers=1024, batch_size=20, vshift=10):
        super(SyncNetInstance, self).__init__()
        self.__S__ = S(num_layers_in_fc_layers=num_layers_in_fc_layers).cuda()
        self.batch_size = batch_size
        self.vshift = vshift
        
    def loadParameters(self, path):
        loaded_state = torch.load(path, map_location=lambda storage, loc: storage)
        self_state = self.__S__.state_dict()
        for name, param in loaded_state.items():
            self_state[name].copy_(param)
            
    def evaluate(self, video_path, audio_path=None, video_fps=25):
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_dir = os.path.join(tmp_dir, 'frames')
            os.makedirs(image_dir)
            subprocess.call(
                f'ffmpeg -y -i "{video_path}" -threads 1 -f image2 {image_dir}/%06d.jpg',
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            if audio_path is None:
                audio_tmp = os.path.join(tmp_dir, 'audio.wav')
                subprocess.call(
                    f'ffmpeg -y -i "{video_path}" -ac 1 -vn -acodec pcm_s16le -ar 16000 "{audio_tmp}"',
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            else:
                audio_tmp = os.path.join(tmp_dir, 'audio.wav')
                shutil.copyfile(audio_path, audio_tmp)

            flist = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
            images = [cv2.imread(f) for f in flist]
            im = np.stack(images, axis=3)
            im = np.expand_dims(im, axis=0)
            im = np.transpose(im, (0, 3, 4, 1, 2))
            imtv = torch.autograd.Variable(torch.from_numpy(im.astype(float)).float())

            sr, audio = wavfile.read(audio_tmp)
            # print(len(images))
            # print(f'audio length: {len(audio) / 16000}', "image length: ", len(images) / video_fps)
            if abs(len(images) / video_fps - len(audio) / 16000) > 0.01:
                print(f'Warning: {video_path} has a frame rate/audio sampling rate mismatch, image: {len(images) / video_fps}, audio: {len(audio) / 16000}')
                return []
            
            mfcc = zip(*python_speech_features.mfcc(audio, sr))
            mfcc = np.stack([np.array(i) for i in mfcc])
            cc = np.expand_dims(np.expand_dims(mfcc, axis=0), axis=0)
            cct = torch.autograd.Variable(torch.from_numpy(cc.astype(float)).float())

            self.__S__.eval()
            batch_size = self.batch_size
            vshift = self.vshift
            
            min_length = min(len(images), len(audio) // 640)
            lastframe = min_length - 5
            im_feat, cc_feat = [], []

            for i in range(0, lastframe, batch_size):
                im_batch = [imtv[:, :, v : v + 5, :, :] for v in range(i, min(lastframe, i + batch_size))]
                im_in = torch.cat(im_batch, 0).cuda()
                im_out = self.__S__.forward_lip(im_in)
                im_feat.append(im_out.data.cpu())

                cc_batch = [cct[:, :, :, v * 4 : v * 4 + 20] for v in range(i, min(lastframe, i + batch_size))]
                cc_in = torch.cat(cc_batch, 0).cuda()
                cc_out = self.__S__.forward_aud(cc_in)
                cc_feat.append(cc_out.data.cpu())

            im_feat = torch.cat(im_feat, 0)
            cc_feat = torch.cat(cc_feat, 0)
            dists = calc_pdist(im_feat, cc_feat, vshift=vshift)
            mdist = torch.mean(torch.stack(dists, 1), 1)
            _, minidx = torch.min(mdist, 0)
            offset =  vshift - minidx
            fdist = np.stack([dist[minidx].numpy() for dist in dists])
            fconf = torch.median(mdist).numpy() - fdist
            fconfm = signal.medfilt(fconf, kernel_size=9)
            np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
            print('Framewise conf: ')
            print(fconfm)
            print(offset)
            print(mdist)
            return fconfm
    
