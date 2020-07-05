# Tracking

Tools for recognize people over the frames.

## [color_tracking](color_tracking.py)

This module contain object used to track persons on a frame from a dictionary.
All of the tracker return a tensor where each line represent an object detected by trt_pose, and each column a tracked person.

### KeypointsColorTracker

This object analyse the color of the r pixels next a given keypoint.
It create a [ROI](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_core/py_basic_ops/py_basic_ops.html#image-roi)
centered on the keypoint and return the ratio of pixels the matching the criteria.

1) Get the range using tools.color_range.HSV_Range

Refer to tools [documentation](../tools/README.md)

2) Create the parametres of the KeypointsColorTracker.

 
 ```python
import cv2
import torch
import numpy as np
from tools.frame_reshaping import ReshapePic1
from tracking.color_tracking import KeypointsColorTracker
from tools.color_range import HSV_Range

color_finder = HSV_Range()  # Instanciation 
cap = cv2.VideoCapture("video.mp4")  # Create a VideoCapture
color_finder(cap)

# Get the color range using the tool
upperBlue = color_finder.getBounds()[0, :, :, 0]
lowerBlue = color_finder.getBounds()[0, :, :, 1]

upperRed = color_finder.getBounds()[1, :, :, 0]
lowerRed = color_finder.getBounds()[1, :, :, 1]

cap = cv2.VideoCapture("video.mp4")
keypoints_tensor = torch.load("tensor.pt")  #  Load the keypoints tensor of this video

keypoints = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow",
             "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee",
             "left_ankle", "right_ankle", "neck"]


nakmuaycoder = {'left_ankle': {"lowerb": lowerBlue, "upperb": upperBlue, "r": 3},
         'right_ankle': {"lowerb": lowerBlue, "upperb": upperBlue, "r": 3},
         'left_hip': {"lowerb": lowerBlue, "upperb": upperBlue, "r": 3},
         'right_hip': {"lowerb": lowerBlue, "upperb": upperBlue, "r": 3}
         }

thongchai = {'right_hip': {"lowerb": lowerRed, "upperb": upperRed, "r": 3},
             'left_hip': {"lowerb": lowerRed, "upperb": upperRed, "r": 3},
             'right_wrist': {"lowerb": lowerRed, "upperb": upperRed, "r": 3},
             'left_wrist': {"lowerb": lowerRed, "upperb": upperRed, "r": 3}
             }


color_tracker = KeypointsColorTracker(keypoints=keypoints, nakmuaycoder=nakmuaycoder, thongchai=thongchai)
rsp = ReshapePic1()

i = 0
while True:
    _, frame = cap.read()
    
    if frame is None:
        break
    frame = rsp(frame)  #  Reshape the frame
    color_tracker(frame=frame, keypoints_tensor=keypoints_tensor[i])
    i += 1

color_tracker.saveProbTable()  # Save the probtab
color_tracker.tracking_result(threshold=0.7, method=0, frame_number=-1)  #Return for each col the row of max prob
color_tracker.tracking_result(threshold=0.7, method=1, frame_number=-1)  # Return for each row the index of maximal prob


``` 


### ClothTracker


## [positional_tracking](tracking/positional_tracking.py)

Track people over the frame using their posture.




