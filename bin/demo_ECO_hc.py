import glob
import os
import pandas as pd
import argparse
import numpy as np
# from PIL import Image
from scipy.misc import imread
import cv2
from eco import ECOTracker

import ipdb as pdb

def main():
    # load videos
    filenames = sorted(glob.glob("./sequences/Crossing/img/*"),
           key=lambda x: int(os.path.basename(x).split('.')[0]))
    # frames = [np.array(Image.open(filename)) for filename in filenames]
    # frames = [cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB) for filename in filenames]
    frames = [imread(filename) for filename in filenames]
    width, height = frames[0].shape[:2]
    if len(frames[0].shape) == 3:
        is_color = True
    else:
        is_color = False
        frames = [frame[:, :, np.newaxis, :] for frame in frames]
    gt_bboxes = pd.read_csv("./sequences/Crossing/groundtruth_rect.txt", sep='\t',
            header=None, names=['xmin', 'ymin', 'width', 'height'])

    # starting tracking
    tracker = ECOTracker(width, height, is_color)
    for idx, frame in enumerate(frames):
        if idx == 0:
            tracker.init(frame, gt_bboxes.iloc[0].values)
        elif idx < len(frames) - 1:
            tracker.update(frame, True)
        else: # last frame
            tracker.update(frame, False)


if __name__ == "__main__":
    main()
