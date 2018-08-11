import glob
import os
import pandas as pd
import argparse
import numpy as np
# from PIL import Image
# from scipy.misc import imread
import cv2
from eco import ECOTracker

import ipdb as pdb

def main():
    # load videos
    filenames = sorted(glob.glob("./sequences/Soccer/img/*"),
           key=lambda x: int(os.path.basename(x).split('.')[0]))
    # frames = [cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB) for filename in filenames]
    frames = [cv2.imread(filename) for filename in filenames]
    width, height = frames[0].shape[:2]
    if len(frames[0].shape) == 3:
        is_color = True
    else:
        is_color = False
        frames = [frame[:, :, np.newaxis, :] for frame in frames]
    gt_bboxes = pd.read_csv("./sequences/Soccer/groundtruth_rect.txt", sep='\t|,',
            header=None, names=['xmin', 'ymin', 'width', 'height'])

    # starting tracking
    tracker = ECOTracker(width, height, is_color)
    for idx, frame in enumerate(frames):
        if idx == 0:
            bbox = gt_bboxes.iloc[0].values
            tracker.init(frame, bbox)
            bbox = (bbox[0]-1, bbox[1]-1,
                    bbox[0]+bbox[2]-1, bbox[1]+bbox[3]-1)
        elif idx < len(frames) - 1:
            bbox = tracker.update(frame, True)
        else: # last frame
            bbox = tracker.update(frame, False)
        # bbox xmin ymin xmax ymax
        frame = cv2.rectangle(frame,
                              (int(bbox[0]), int(bbox[1])),
                              (int(bbox[2]), int(bbox[3])),
                              (0, 255, 0),
                              1)
        cv2.imshow("corssing", frame)
        cv2.waitKey(30)

if __name__ == "__main__":
    main()
