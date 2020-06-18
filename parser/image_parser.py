"""
ImageParser.
Run trt_pose on a single image

All of those object inherit from parser._ImageParser class

"""
import cv2
import numpy as np
from parser import _ImageParser

class ReshapePic1(object):
    """ Resize and crop the center of the picture
     - Good for tv video where characters a @ the middle of the image"""

    def __init__(self, shape):
        """
        :param shape: tuple, destination shape
        """
        self.shape = shape

    def __call__(self, array):
        """
        :param array: numpy array, an image to resize
        :return: numpy array, an image resized
        """
        w, h, _ = array.shape
        if w >= h:
            w = self.shape[0] * w // h
            h = self.shape[1]
        else:
            h = self.shape[1] * h // w
            w = self.shape[0]
        array = cv2.resize(array, (h, w))
        first_row = (w - self.shape[0]) // 2
        first_col = (h - self.shape[1]) // 2
        return array[first_row: first_row + self.shape[0], first_col: first_col + self.shape[1], :]


class ReshapePic2(object):
    """Resize the picture and add black bands"""

    def __init__(self, shape):
        """
        :param shape: tuple, destination shape
        """
        self.shape = shape

    def __call__(self, array):
        """
        :param array: numpy array, an image to resize
        :return: numpy array, an image resized
        """
        w, h, _ = array.shape
        if w <= h:
            w = self.shape[0] * w // h
            h = self.shape[1]
        else:
            h = self.shape[1] * h // w
            w = self.shape[0]
        array = cv2.resize(array, (h, w))
        frame = np.zeros((self.shape[0], self.shape[1], 3), dtype="uint8")
        frame[0:w, 0:h, :] += array
        return frame


class ImageParser(_ImageParser):
    """ Parse an image"""

    def __init__(self, trt_model, parse_objects):
        """
        :param trt_model: a torch model optimized for tensorRT
        :param parse_objects: a trt_pose.parse_objects.ParseObjects object
        """
        super().__init__(trt_model, parse_objects)

    def __call__(self, frame, max_detection=100):
        """Apply _parse_image method
        :param frame: 3 dimensional np.ndarray
        :param max_detection: Maximal number of person detected
        :return: a tensor of shape (number of person detected, number of points, 2) Chanel 0 : y; Chanel 1: x
        """
        peak = self._parse_image(frame, max_detection)
        return peak
