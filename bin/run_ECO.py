import numpy as np
import time
from eco import ECOTracker
import cv2
import glob
import os
from tqdm import tqdm
from PIL import Image


def run_ECO(seq, rp, saveimage):
    x = seq.init_rect[0]
    y = seq.init_rect[1]
    w = seq.init_rect[2]
    h = seq.init_rect[3]

    frames = [np.array(Image.open(filename)) for filename in seq.s_frames]
    # frames = [cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB) for filename in seq.s_frames]
    if len(frames[0].shape) == 3:
        is_color = True
    else:
        is_color = False
        frames = [frame[:, :, np.newaxis] for frame in frames]
    tic = time.clock()
    # starting tracking
    tracker = ECOTracker(is_color)
    res = []
    for idx, frame in enumerate(frames):
        if idx == 0:
            bbox = (x, y, w, h)
            tracker.init(frame, bbox)
            bbox = (bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3])
        elif idx < len(frames) - 1:
            bbox = tracker.update(frame, True)
        else: # last frame
            bbox = tracker.update(frame, False)
        res.append((bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]))
    duration = time.clock() - tic
    result = {}
    result['res'] = res
    result['type'] = 'rect'
    result['fps'] = round(seq.len / duration, 3)
    return result

