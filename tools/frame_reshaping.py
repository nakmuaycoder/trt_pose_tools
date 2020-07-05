"""Tools for reshape the data"""

import cv2
import numpy as np


class _Rsizer(object):
    def __init__(self, shape):
        self.shape = shape

    def __repr__(self):
        return "Shape ({}, {}, {})".format(self.shape[0], self.shape[1], self.shape[2])



class ReshapePic1(_Rsizer):
    """ Resize and crop the center of the picture
     - Good for tv video where characters a @ the middle of the image"""

    def __init__(self, shape=(224, 224, 3)):
        """
        :param shape: tuple, destination shape
        """
        super().__init__(shape=shape)

    def _reshape_pic1(self, array):
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
        array = cv2.resize(array, (h, w), interpolation=cv2.INTER_AREA)
        first_row = (w - self.shape[0]) // 2
        first_col = (h - self.shape[1]) // 2
        return array[first_row: first_row + self.shape[0], first_col: first_col + self.shape[1], :]

    def __call__(self, array):
        """
        :param array: numpy array, an image to resize
        :return: numpy array, an image resized
        """
        return self._reshape_pic1(array)



class ReshapePic2(_Rsizer):
    """Resize the picture and add black bands"""

    def __init__(self, shape=(224, 224, 3)):
        """
        :param shape: tuple, destination shape
        """
        super().__init__(shape=shape)

    def _reshape_pic2(self, array):
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
        array = cv2.resize(array, (h, w), interpolation=cv2.INTER_AREA)
        frame = np.zeros((self.shape[0], self.shape[1], 3), dtype="uint8")
        frame[0:w, 0:h, :] += array
        return frame

    def __call__(self, array):
        """
        :param array: numpy array, an image to resize
        :return: numpy array, n image resized
        """
        return self._reshape_pic2(array)


class ROI(ReshapePic1, ReshapePic2):
    """Return a special zone of a picture and return it with the desired shape"""
    def __init__(self, shape=(224, 224, 3)):
        """Instanciate the final shape of the output"""
        _Rsizer.__init__(shape)

    def __call__(self, array, point1, point2, method=1):
        """
         Apply _reshape_pic1 or _reshape_pic2 to a ROI
        :param array:  numpy array, an image to resize
        :param point1: coordinate (x: int, y: int) of the first corner of the ROI
        :param point2: coordinate (x: int, y: int) of a second corner of the ROI
        :param method: int 1 or 2, apply reshape the picture using ReshapePic1 or ReshapePic2
        :return:
        """
        min_x = min(point1[0], point2[0])
        max_x = max(point1[0], point2[0])
        min_y = min(point1[1], point2[1])
        max_y = max(point1[1], point2[1])

        array = array[min_x:max_x, min_y:max_y, :]

        if method == 1:
            return self._reshape_pic1(array)
        elif method == 2:
            return self._reshape_pic2(array)
