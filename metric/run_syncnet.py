#!/usr/bin/python
#-*- coding: utf-8 -*-

import time, pdb, argparse, subprocess, pickle, os, gzip, glob

from omegaconf import OmegaConf

from SyncNetInstance import *

def syncnet_run(s, video_path, video_reference, data_dir):
    # ==================== PARSE ARGUMENT ====================
    opt = OmegaConf.create(
        { 
        "data_dir": data_dir,
        "videofile": video_path,
        "reference": video_reference,
        "batch_size": 20,
        "vshift": 15,
        }
    )

    opt['avi_dir'] = os.path.join(opt.data_dir,'pyavi')
    opt['tmp_dir'] = os.path.join(opt.data_dir,'pytmp')
    opt['work_dir'] = os.path.join(opt.data_dir,'pywork')
    opt['crop_dir'] = os.path.join(opt.data_dir,'pycrop')

    # ==================== LOAD MODEL AND FILE LIST ====================

    flist = glob.glob(os.path.join(opt.crop_dir,opt.reference,'0*.avi'))
    flist.sort()

    # ==================== GET OFFSETS ====================

    offsets = []
    dists = []
    confs = []
    minvals = []
    for idx, fname in enumerate(flist):
        offset, conf, dist, minval, fconm = s.evaluate(opt,videofile=fname)
        offsets.append(offset)
        dists.append(dist)
        confs.append(conf)
        minvals.append(minval)
    if len(flist) == 1:
        return dists, confs[0], minvals[0], offsets[0], fconm
    else:
        return 0, 0, 0, 0, 0
        
    # # ==================== PRINT RESULTS TO FILE ====================

    # with open(os.path.join(opt.work_dir,opt.reference,'activesd.pckl'), 'wb') as fil:
    #     pickle.dump(dists, fil)
