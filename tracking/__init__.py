"""
Tracking package:

Mother class of all the tracker

========================================================================================================================
For prod with torch,
UNCOMMENT line 11, 47, 49 and comment line 13, 48, 50
========================================================================================================================
"""

# import torch
import cv2
import numpy as np


class _Tracker(object):
    """Mother class of the tracker

    - tracking result is a tensor of shape: (1, number de person to track)
    Ex track 2 persons: (id_person1, id_person2)
    where id_personK is the line number of this person in the parser result
    """

    def __init__(self, keypoints, **kwargs):
        """
        :param keypoints: dict, couples keypoints:col in the tensor
        :param kwargs: each argument is a dict: couples person_name:id
        """
        self.dict_keypoints = {j: i for i, j in enumerate(keypoints)}
        self.kwargs = kwargs
        self.dict_object = {i: j for i, j in enumerate(kwargs)}

    def _show_tracking(self, keypoint_tensor, frame, tracking_result):
        """
        Show the result:
        Write rectangle in the frame
        :param keypoint_tensor: the tensor with coordinate of the detected keypoints
        :param frame: the frame
        :param tracking_result: a tensor num_pers
        :return:
        """
        for element, detected in enumerate(tracking_result):
            if detected in self.dict_object.keys():
                detected = int(detected)
                name = self.dict_object[element]
                keypoint = keypoint_tensor[0][detected]
                #y_min, y_max = int(torch.min(keypoint[:, 1])), int(torch.max(keypoint[:, 1]))
                y_min, y_max = int(np.nanmin(keypoint[:, 1])), int(np.nanmax(keypoint[:, 1]))
                #x_min, x_max = int(torch.min(keypoint[:, 0])), int(torch.max(keypoint[:, 0]))
                x_min, x_max = int(np.nanmin(keypoint[:, 0])), int(np.nanmax(keypoint[:, 0]))
                cv2.rectangle(img=frame, pt1=(x_min, y_min), pt2=(x_max, y_max), color=(0, 0, 255), thickness=1)
                cv2.putText(img=frame, text=name, org=(x_min, y_min), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.5,
                            color=(0, 0, 255), thickness=2)

        cv2.imshow("Tracking", frame)
        cv2.waitKey()

    def __add__(self, other):
        """Adding two tracker create a tracker with the 2 dict"""
        if self.dict_keypoints == other.dict_keypoints:
            return self.__class__(keypoints=self.dict_keypoints.keys(), **self.kwargs, **other.kwargs)

    def __repr__(self):
        return str(self.dict_object)

    def __contains__(self, item):
        """ True if item is parametres of tracking"""
        return any(_ not in self.kwargs.items() for _ in item.items())

# ===================================================  Test Class  =====================================================

keypoints = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow",
            "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee",
            "left_ankle", "right_ankle", "neck"]


image = cv2.imread("C:\\Users\\rose_\\Desktop\\olivier\\trt_pose_tools\\data\\20200315_124340.jpg")
x, y, _ = image.shape

x //= 8
y //= 8

image = cv2.resize(image, (y, x), interpolation=cv2.INTER_AREA)

# (frames, max_detection, points, 2)
tracking_result = np.asarray([1, 0])
keypoint_tensor = np.zeros(shape=(1, 10, 9, 2)) * np.nan

# first person
keypoint_tensor[0][0][:, 0] = np.asarray([100, 172, 210, 135, 150, np.nan, np.nan, np.nan, np.nan])
keypoint_tensor[0][0][:, 1] = np.asarray([292, 267, 269, 270, 300, np.nan, np.nan, np.nan, np.nan])

# second person
keypoint_tensor[0][1][:, 0] = np.asarray([254, 238, 272, 295, 231, 353, 370, 331, 275])
keypoint_tensor[0][1][:, 1] = np.asarray([342, 331, 330, 341, 388, 406, 495, 387, 386])

olive = _Tracker(keypoints=keypoints, olive={"a": 1, "b": 2, "c": 3})
nana = _Tracker(keypoints=keypoints, nana={"a": 1, "b": 2, "c": 3})

nous = olive + nana

#nous._show_tracking(frame=image, keypoint_tensor=keypoint_tensor, tracking_result=tracking_result)

