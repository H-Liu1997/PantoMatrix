#!/usr/bin/python
#-*- coding: utf-8 -*-

import time, pdb, argparse, subprocess, pickle, os, gzip, glob

from SyncNetInstance import *

# ==================== PARSE ARGUMENT ====================

parser = argparse.ArgumentParser(description = "SyncNet");
parser.add_argument('--initial_model', type=str, default="data/syncnet_v2.model", help='');
parser.add_argument('--batch_size', type=int, default='20', help='');
parser.add_argument('--vshift', type=int, default='15', help='');
parser.add_argument('--data_dir', type=str, default='data/work', help='');
parser.add_argument('--videofile', type=str, default='', help='');
parser.add_argument('--reference', type=str, default='', help='');
opt = parser.parse_args();

setattr(opt,'avi_dir',os.path.join(opt.data_dir,'pyavi'))
setattr(opt,'tmp_dir',os.path.join(opt.data_dir,'pytmp'))
setattr(opt,'work_dir',os.path.join(opt.data_dir,'pywork'))
setattr(opt,'crop_dir',os.path.join(opt.data_dir,'pycrop'))


# ==================== LOAD MODEL AND FILE LIST ====================

s = SyncNetInstance(vshift=opt.vshift, batch_size=opt.batch_size);
s.loadParameters(opt.initial_model);
print("Model %s loaded."%opt.initial_model);
conf = s.evaluate(video_path="/home/weili/haiyang/syncnet_python/outputs/tmp2/t.mp4", audio_path="/home/weili/haiyang/PantoMatrix/HDTF/cache_audio_v4/RD_Radio1_000_001.wav", video_fps=24)
print(conf, sum(conf)/len(conf))
