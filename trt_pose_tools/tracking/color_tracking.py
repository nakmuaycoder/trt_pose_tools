"""
Use color filter to recognize person

"""

from . import _ArgumentsParser
import cv2
import numpy as np
import torch


class KeypointsColorTracker(_ArgumentsParser):
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

        result = result.unsqueeze(0)  # Add a channel
        if not hasattr(self, "_prob"):
            self._prob = result
        else:
            self._prob = torch.cat([self._prob, result], dim=0)

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
                roi = self._roi(frame=hsv, upper_left_corner=(max(int(y) - r, 0), max(int(x) - r, 0)),
                                bottom_right_corner=(min(int(y) + r, hsv.shape[0]), min(int(x) + r, hsv.shape[1])))

                # Apply mask to ROI and count the percent of pixels that match
                mask = cv2.inRange(roi, lowerb=lowerb, upperb=upperb)
                mask = np.where(mask > 0, 1, 0).sum() / np.prod(mask.shape)
                prob += mask / npoints
        return prob


keypoints = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow",
             "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee",
             "left_ankle", "right_ankle", "neck"]


class ClothTracker(KeypointsColorTracker):
    """"""

    def __init__(self, keypoints, **kwargs):
        keypoints += ["short", "tshirt"]
        super().__init__(keypoints, **kwargs)

    def _get_cloth_range(self, keypoints):
        """
        Range for roi
        :param keypoints:
        :return:
        """
        left_shoulder = self.dict_keypoints['left_shoulder']
        right_shoulder = self.dict_keypoints['right_shoulder']
        neck = self.dict_keypoints['neck']
        right_hip = self.dict_keypoints['right_hip']
        left_hip = self.dict_keypoints['left_hip']
        right_knee = self.dict_keypoints['right_knee']
        left_knee = self.dict_keypoints['left_knee']

        tshirt = [(np.nanmin(keypoints[left_shoulder, 1], keypoints[right_shoulder, 1], keypoints[neck, 1],
                       keypoints[right_hip, 1], keypoints[right_hip, 1]),
                   np.nanmin(keypoints[left_shoulder, 0], keypoints[right_shoulder, 0], keypoints[neck, 0]),
                   keypoints[right_hip, 0], keypoints[right_hip, 0]),
                  (np.nanmax(keypoints[left_shoulder, 1], keypoints[right_shoulder, 1], keypoints[neck, 1],
                       keypoints[right_hip, 1], keypoints[right_hip, 1]),
                   np.nanmax(keypoints[left_shoulder, 0], keypoints[right_shoulder, 0], keypoints[neck, 0],
                       keypoints[right_hip, 0], keypoints[right_hip, 0]))]
        
        short = [(np.nanmin(keypoints[left_hip, 1], keypoints[right_hip, 1], keypoints[left_knee, 1], keypoints[right_knee, 1]),
                 np.nanmin(keypoints[left_hip, 0], keypoints[right_hip, 0], keypoints[left_knee, 0], keypoints[right_knee, 0])),
                 (np.nanmax(keypoints[left_hip, 1], keypoints[right_hip, 1], keypoints[left_knee, 1], keypoints[right_knee, 1]),
                  np.nanmax(keypoints[left_hip, 0], keypoints[right_hip, 0], keypoints[left_knee, 0], keypoints[right_knee, 0]))]

        return {'tshirt': tshirt, 'short': short}

    def _track_single_person(self, keypoints, frame, dictio):
        """
        Test if 1 person detectd by trt_pose match with the dict.
        Create a ROI, of diagonal length r and center the keypoint returned by the model.
        Apply a mask to isolate the pixels between lowerb and upperb.
        :param keypoints: tensor of shape (1, npoints,2) of points coordinate
        :param frame: np.ndarray
        :param dictio: single person list of points color and radius (1 element of self.kwargs): ex
        {"tshirt":{"lowerb":np.asarray([110, 50, 50]),upperb":np.asarray([130, 255, 255])}, short:}
        :return: proba, if cant test np.nan, else return a proba
        """
        # Convert the frame in hsv
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        prob = np.nan

        npoints = len(dictio.keys())  # number of point to test
        for point in dictio:
            lowerb = dictio[point]["lowerb"]
            upperb = dictio[point]["upperb"]

            rge = self._get_cloth_range(keypoints=keypoints)[point]

            if not np.isnan(rge[0][0]) and not np.isnan(rge[0][1]) and not np.isnan(rge[1][0]) and not np.isnan(rge[1][1]):
                # Create the ROI
                roi = self._roi(frame=hsv, upper_left_corner=rge[0], bottom_right_corner=rge[1])

                # Apply mask to ROI and count the percent of pixels that match
                mask = cv2.inRange(roi, lowerb=lowerb, upperb=upperb)
                mask = np.where(mask > 0, 1, 0).sum() / np.prod(mask.shape)
                prob += mask / npoints
        return prob
