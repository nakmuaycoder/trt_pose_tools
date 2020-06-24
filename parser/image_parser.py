"""
ImageParser.
Run trt_pose on a single image

All of those object inherit from parser._ImageParser class

"""
import cv2
import numpy as np
from . import _ImageParser

class ImageParser(_ImageParser):
    """ Parse an image"""

    def __init__(self, trt_model, parse_objects, points=18):
        """
        :param trt_model: a torch model optimized for tensorRT
        :param parse_objects: a trt_pose.parse_objects.ParseObjects object
        :param points: number of points detected by the model
        """
        super().__init__(trt_model, parse_objects, points)

    def __call__(self, frame, max_detection=100):
        """Apply _parse_image method
        :param frame: 3 dimensional np.ndarray
        :param max_detection: Maximal number of person detected
        :return: a tensor of shape (number of person detected, number of points, 2) Chanel 0 : y; Chanel 1: x
        """
        peak = self._parse_image(frame, max_detection)
        return peak
