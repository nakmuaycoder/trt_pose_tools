# Tracking

Tools for recognize people over the frames.

## [color_tracking](color_tracking.py)

This module contain object used to track persons on a frame from a dictionary.
All of the tracker return a tensor where each line represent an object detected by trt_pose, and each column a tracked person.

### KeypointsColorTracker

This object analyse the color of the r pixels next a given keypoint.
It create a [ROI](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_core/py_basic_ops/py_basic_ops.html#image-roi)
centered on the keypoint and return the ratio of pixels the matching the criteria.

 
 ```python
import cv2
import torch
import numpy as np
from parser.tools import ReshapePic1
from tracking.color_tracking import KeypointsColorTracker

cap = cv2.VideoCapture("seq1.mp4")  #  Open a VideoCapture object
keypoints_tensor = torch.load("tensor.pt")  #  Load the keypoints tensor of this video

keypoints = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow",
             "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee",
             "left_ankle", "right_ankle", "neck"]

nakmuaycoder = {'left_ankle': {"lowerb": np.asarray([110, 50, 50]), "upperb": np.asarray([130, 255, 255]), "r": 5},
         'right_ankle': {"lowerb": np.asarray([110, 50, 50]), "upperb": np.asarray([130, 255, 255]), "r": 5},
         'left_wrist': {"lowerb": np.asarray([110, 50, 50]), "upperb": np.asarray([130, 255, 255]), "r": 5},
         'right_wrist': {"lowerb": np.asarray([110, 50, 50]), "upperb": np.asarray([130, 255, 255]), "r": 5},
         'left_hip': {"lowerb": np.asarray([110, 50, 50]), "upperb": np.asarray([130, 255, 255]), "r": 5},
         'right_hip': {"lowerb": np.asarray([110, 50, 50]), "upperb": np.asarray([130, 255, 255]), "r": 5}
         }

thongchai = {'right_hip': {"lowerb": np.asarray([161, 155, 84]), "upperb": np.asarray([179, 255, 255]), "r": 3},
             'left_hip': {"lowerb": np.asarray([161, 155, 84]), "upperb": np.asarray([179, 255, 255]), "r": 3},
             'right_wrist': {"lowerb": np.asarray([161, 155, 84]), "upperb": np.asarray([179, 255, 255]), "r": 3},
             'left_wrist': {"lowerb": np.asarray([161, 155, 84]), "upperb": np.asarray([179, 255, 255]), "r": 3}
             }

tracker = KeypointsColorTracker(keypoints=keypoints, nakmuaycoder=nakmuaycoder, thongchai=thongchai)
rsp = ReshapePic1((224, 224, 3))


_, frame = cap.read()
cap.release()

frame = rsp(frame)  #  Reshape the frame
tracker(frame=frame, keypoints_tensor=keypoints_tensor[0])
``` 


### ClothTracker
