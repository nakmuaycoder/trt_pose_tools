"""
ImageParser.
Run trt_pose on a single image

All of those object inherit from parser._ImageParser class

"""
import cv2
import numpy as np
from . import _ImageParser
import torch

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

        max_batch_size = self.max_batch_size
        out = torch.zeros((0, max_detection, self.points, 2))

        try:
            batch_size = len(frame)
        except:
            frame = [frame]
            batch_size = 1
        
        nbatch = batch_size // max_batch_size
        rest = batch_size % max_batch_size

        for i in range(nbatch):
            peak = self._parse_image(frame[i * max_batch_size:(i + 1) * max_batch_size], max_detection)
            out = torch.cat([out, peak], dim=0)

        if rest != 0:
            peak = self._parse_image(frame[-rest:], max_detection)
            out = torch.cat([out, peak], dim=0)

        return out
