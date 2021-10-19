"""
image_parser.py
run inference on a single image

author : nakmuaycoder
date : 10/2021
"""

import numpy as np
import torch
import torchvision.transforms as transforms
import PIL.Image
import cv2
import torch2trt
from trt_pose_tools.model import load_optz, get_parse_objects


class ImageParser:
    """
    ImageParser object, perform trt_pose, pose estimation to a single frame
    """
    def __init__(self, max_detection=100, reshape_frame=None):
        """
        Instantiation of the parser.
        - load the model
        max_detection : int, number of detected persons
        reshape_frame : callable, reshaping the picture
        """
        self.__trt_model = load_optz(1)
        self.__parse_objects = get_parse_objects()
        self.__points = 18
        self.set_max_detection(max_detection)
        self.set_reshape_frame(reshape_frame)
    
    def get_max_detection(self):
        """
        return the number max of detection
        """
        return self.__max_detection

    def set_max_detection(self, max_detection):
        """
        Change the number max of detection
        """
        if not isinstance(max_detection, int):
            raise TypeError("max_detection must be integer")
        self.__max_detection = max_detection

    def set_reshape_frame(self, reshape_frame):
        """
        set a reshape_fram
        """
        if reshape_frame is None:
            reshape_frame = lambda x: x
        if not callable(reshape_frame):
            raise AttributeError("reshape_frame must be callable")
        self.__reshape_frame = reshape_frame

    def __preprocess(self, image):
        """
        Image preprocessing.
        image : np.ndarray of type uint8 and shape (224, 224, 3)
        """
        device = torch.device('cuda')
        mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
        std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = PIL.Image.fromarray(image)
        image = transforms.functional.to_tensor(image).to(device)
        image.sub_(mean[:, None, None]).div_(std[:, None, None])
        return image[None, ...]

    def __parse_image(self, frame, max_detection, reshape_frame):
        """
        Parse a single image
        frame : np.ndarray of type uint8 and shape (224, 224, 3)
        max_detection : int, number of detected persons
        reshape_frame : callable, reshaping the picture
        """
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a numpy array of shape (224, 224, 3)")
        frame = reshape_frame(frame)
        if frame.shape != (224, 224, 3):
             raise ValueError("frame must be a numpy array of shape (224, 224, 3)")

        data = self.__preprocess(frame)  # Frame preprocessing
        cmap, paf = self.__trt_model(data)  # Inference
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        counts, objects, normalized_peaks = self.__parse_objects(cmap, paf)
        _ , _, point = objects.shape
        out = torch.zeros((max_detection, point, 2)) * np.nan  # Output instantiation      
        for obj_index in range(min(counts[0], max_detection)):
            obj = objects[0, obj_index]
            out[obj_index] = torch.Tensor([[normalized_peaks[0, i, _, 0], normalized_peaks[0, i, _, 1]] if _ >= 0 else [np.nan, np.nan] for i, _ in  enumerate(obj) ])
        out *= 224
        return out

    def __call__(self, frame):
        """
        Parse a frame
        """
        return self.__parse_image(frame, self.__max_detection, self.__reshape_frame)
    
    def get_points(self):
        """
        Return number of points
        """
        return self.__points
