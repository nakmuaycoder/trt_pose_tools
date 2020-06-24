"""
Use color filter to recognize person

"""

from . import _Tracker
import cv2
import numpy as np
import torch


class KeypointsColorTracker(_Tracker):
    """ Tracking using the color in the area near a keypoints.
    Identification require the frame, run only using 1 frame
    ex for boxing: hand :blue
    hips"""

    def __init__(self, keypoints, **kwargs):
        # Test if all keywords are found in **kwargs
        try:
            for tracked_person in kwargs:
                if not set(kwargs[tracked_person].keys()).issubset((keypoints)):
                    # test for each person object to detect are in keypoints
                    raise ValueError(tracked_person + " have unknown keypoints.")

            super().__init__(keypoints=keypoints, **kwargs)

        except ValueError as v:
            print("ValueError: ", v)

    def __call__(self, frame, keypoints_tensor):
        """
        Select a ROI near a keypoint and apply color mask
        :param frame: (np.ndarray) the frame
        :param keypoints_tensor: (torch.Tensor) the output tensor or the model (max_detection, number of points, 2)
        :return:
        """
        max_detection, _, _ = keypoints_tensor.shape
        # tensor 1 line per detected person/ 1 col per tracked person
        result = torch.zeros((max_detection, len(self.dict_object.keys()))) * np.nan

        for col_index, name in self.dict_object.items():
            # Fill result col per col (Loop over the person to track)
            points = self.kwargs[name]
            prob = torch.Tensor([self._track_single_person(keypoints=_, frame=frame, dictio=points) for _ in keypoints_tensor])
            result[:, col_index] = prob

        return result

    def _roi(self, upper_left_corner, bottom_right_corner, frame):
        """Create the ROI to apply filter"""
        x1, y1 = upper_left_corner
        x2, y2 = bottom_right_corner
        return frame[x1:x2, y1:y2]

    def _track_single_person(self, keypoints, frame, dictio):
        """
        Test if 1 person detectd by trt_pose match with the dict.
        Create a ROI, of diagonal length radius and center the keypoint returned by the model.
        Apply a mask to isolate the pixels between lowerb and upperb.
        :param keypoints: tensor of shape (1, npoints,2) of points coordinate
        :param frame: np.ndarray
        :param dictio: single person list of points color and radius (1 element of self.kwargs): ex
        {"left_ankle":{"lowerb":np.asarray([110, 50, 50]),upperb":np.asarray([130, 255, 255]), r:10}}
        :return: proba, if cant test np.nan, else return a proba
        """
        # Convert the frame in hsv
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        prob = np.nan

        npoints = len(dictio.keys())  # number of point to test
        for point in dictio:
            r = dictio[point]["r"]
            lowerb = dictio[point]["lowerb"]
            upperb = dictio[point]["upperb"]

            # point coordinate in the tensor
            y = keypoints[self.dict_keypoints[point], 0]
            x = keypoints[self.dict_keypoints[point], 1]

            # create ROI
            if not torch.isnan(x) or not torch.isnan(y):
                if np.isnan(prob):
                    # initialize the proba to 0
                    prob = 0

                # Create the ROI
                roi = self._roi(frame=hsv, upper_left_corner=(max(int(x) - r, 0), max(int(y) - r, 0)),
                                bottom_right_corner=(min(int(x) + r, hsv.shape[0]), min(int(y) + r, hsv.shape[1])))

                # Apply mask to ROI and count the percent of pixels that match
                mask = cv2.inRange(roi, lowerb=lowerb, upperb=upperb)
                mask = np.where(mask > 0, 1, 0).sum() / np.prod(mask.shape)
                prob += mask / npoints
        return prob
