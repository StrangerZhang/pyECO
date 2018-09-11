import glob
import os
import pandas as pd
import argparse
import numpy as np
import cv2
from eco import ECOTracker
from PIL import Image

import argparse

def main(video_dir):
    # load videos
    filenames = sorted(glob.glob(os.path.join(video_dir, "*.jpg")),
           key=lambda x: int(os.path.basename(x).split('.')[0]))
    # frames = [cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB) for filename in filenames]
    frames = [np.array(Image.open(filename)) for filename in filenames]
    height, width = frames[0].shape[:2]
    if len(frames[0].shape) == 3:
        is_color = True
    else:
        is_color = False
        frames = [frame[:, :, np.newaxis] for frame in frames]
    gt_bboxes = pd.read_csv(os.path.join(video_dir, "groundtruth.txt"), sep='\t|,| ',
            header=None, names=['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4'],
            engine='python')

    title = video_dir.split('/')[-1]
    # starting tracking
    tracker = ECOTracker(is_color)
    for idx, frame in enumerate(frames):
        if idx == 0:
            bbox = gt_bboxes.iloc[0].values
            xmin = np.min(bbox[::2])
            xmax = np.max(bbox[::2])
            ymin = np.min(bbox[1::2])
            ymax = np.max(bbox[1::2])
            bbox = (xmin, ymin, xmax-xmin+1, ymax-ymin+1)
            tracker.init(frame, bbox)
            bbox = (xmin, ymin, xmax, ymax)
        elif idx < len(frames) - 1:
            bbox = tracker.update(frame, True)
        else: # last frame
            bbox = tracker.update(frame, False)
        # bbox xmin ymin xmax ymax
        frame = cv2.rectangle(frame,
                              (int(bbox[0]-1), int(bbox[1]-1)),
                              (int(bbox[2]-1), int(bbox[3]-1)),
                              (0, 255, 0),
                              2)
        gt_bbox = gt_bboxes.iloc[idx].values
        xmin = np.min(gt_bbox[::2])
        xmax = np.max(gt_bbox[::2])
        ymin = np.min(gt_bbox[1::2])
        ymax = np.max(gt_bbox[1::2])
        gt_bbox = (xmin, ymin, xmax, ymax)
        frame = frame.squeeze()
        frame = cv2.rectangle(frame,
                              (int(gt_bbox[0]-1), int(gt_bbox[1]-1)), # 0-index
                              (int(gt_bbox[2]-1), int(gt_bbox[3]-1)),
                              (255, 0, 0),
                              1)
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.putText(frame, str(idx), (5, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
        cv2.imshow(title, frame)
        cv2.waitKey(30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, default='../sequences/Crossing/')
    args = parser.parse_args()
    main(args.video_dir)
