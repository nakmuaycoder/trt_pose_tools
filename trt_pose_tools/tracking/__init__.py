"""
Tracking package:

Mother class of all the tracker
"""

import torch
import cv2
import numpy as np
import os

class _Tracker(object):
    """"""
    def __init__(self):
        pass

    def getProbTable(self):
        """
        Return the prob table
        :return:
        """
        if hasattr(self, "_prob"):
            return self._prob
        else:
            raise AttributeError("The prabtab must be generate at least on time before call it")

    def saveProbTable(self, path):
        """
        Save _prob tensor
        :param path:
        """
        p = os.path.join(path + "prob.pt")
        if not hasattr(self, "_prob"):
            raise TypeError("No probTable to save")

        if not os.path.exists(path):
            raise ValueError("Invalid path")

        if os.path.exists(p):
            raise ValueError("path: " + p + " already exists")
        torch.save(self._prob, p)
        print("probtab saved @", p)


    def _getMaxIndex0(self, threshold, frame_number):
        """
        Return the index of most probable detected object
        :param threshold:
        :return: the row number of the max of each column of self._prob
        """
        prob = self._prob[frame_number]
        mx = np.nanmax(prob, axis=0)
        mx = np.where(mx > threshold, mx, np.nan)
        return np.asarray([np.where(prob[:, _] == mx[_]) for _ in range(mx.shape[0])]).flatten()

    def _getMaxIndex1(self, threshold, frame_number):
        """
        Affect to each row the max of _prob.
        A person can be recognize more than 1 time
        :param threshold:
        :return:
        """
        prob = self._prob[frame_number]
        mx = np.nanmax(prob, axis=1)  # Row max
        mx = np.where(mx > threshold, mx, np.nan)  # Keep mx value over threshold
        res = []
        for i in range(mx.shape[0]):
            try:
                res += [np.where(prob[i] == mx[i])[0][0]]
            except:
                res += [np.nan]
        res = np.asarray(res)
        return res

    def tracking_result(self, threshold=0.7, method=0, frame_number=-1):
        """
        Return the index of the max value for each row of the result tensor
        :param threshold: minimum bound
        :param frame_number: the frame number to analyse the result
        :return:
        """
        if not hasattr(self, "_prob"):
            raise AttributeError("Probtable is missing, generate a tracking before run this method")

        if frame_number > self._prob.shape[0]:
            raise ValueError("frame_number must be < pronTab.shape[0]")

        if threshold > 1 or threshold < 0:
            raise ValueError("threshold must be between 0 and 1")

        if method == 1:
            return self._getMaxIndex1(threshold=threshold, frame_number=frame_number)
        elif method == 0:
            return self._getMaxIndex0(threshold=threshold, frame_number=frame_number)
        else:
            return None

class _RecursiveTracker(_Tracker):
    """Class of tracker:
    tracking is based on the similarity between 2 consecutive frames"""

    def __init__(self):
        """Create a dict.
        the key is the index of the person in the current frame, the value is the index in the initial frame"""
        self.to_initial = {_: _ for _ in range(100)}

    def initial_index(self, current_result):
        """
        Return the index of the first frame
        :param current_result:
        :return:
        """
        res = {}
        for idx, val in enumerate(current_result):
            if val in self.to_initial.keys():
                res[idx] = self.to_initial[val]
            else:
                res[idx] = np.nan
        self.to_initial = res



testRecursive = _RecursiveTracker()
testRecursive.initial_index(np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 0]))




class _ArgumentsParser(_Tracker):
    """Tracker that require arguments

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

    def __add__(self, other):
        """Adding two tracker create a tracker with the 2 dict"""
        if self.dict_keypoints == other.dict_keypoints:
            return self.__class__(keypoints=self.dict_keypoints.keys(), **self.kwargs, **other.kwargs)

    def __repr__(self):
        return str(self.dict_object)

    def __contains__(self, item):
        """ True if item is parametres of tracking"""
        return any(_ not in self.kwargs.items() for _ in item.items())

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
                y_min, y_max = int(torch.min(keypoint[:, 1])), int(torch.max(keypoint[:, 1]))
                #y_min, y_max = int(np.nanmin(keypoint[:, 1])), int(np.nanmax(keypoint[:, 1]))
                x_min, x_max = int(torch.min(keypoint[:, 0])), int(torch.max(keypoint[:, 0]))
                #x_min, x_max = int(np.nanmin(keypoint[:, 0])), int(np.nanmax(keypoint[:, 0]))
                cv2.rectangle(img=frame, pt1=(x_min, y_min), pt2=(x_max, y_max), color=(0, 0, 255), thickness=1)
                cv2.putText(img=frame, text=name, org=(x_min, y_min), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.5,
                            color=(0, 0, 255), thickness=2)

        cv2.imshow("Tracking", frame)
        cv2.waitKey()
