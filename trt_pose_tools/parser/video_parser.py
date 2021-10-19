"""
VideoParser.
Parse all of the frame from a video.

VideoParsers retrun a tensor of shape (number of frames, number of detection, number of points, 2) 

Creation of new parser:

- Create a class that inherite from VideoCaptureParser and add the following line in the init of your class :
super().__init__(video_capture=self.__open_videocapture(video_url), reshape_frame=reshape_frame, max_detection=max_detection)

- Overwrite __open_videocapture method

author : nakmuaycoder
date : 10/2021
"""

import numpy as np
import os
import cv2
import pafy
from .image_parser import ImageParser
import torch
import tqdm

class VideoCaptureParser(ImageParser):
    """
    This class parse a cv2.VideCapture object.
    """
    def __init__(self, video_capture, reshape_frame=None, max_detection=100):
        super().__init__(max_detection,reshape_frame)
        self.__set_video_capture(video_capture)

    def __set_video_capture(self, video_capture):
        """
        Set a VideoCapture
        """
        if hasattr(self, "__video_capture"):
            self.__video_capture.release()
        if not isinstance(video_capture, cv2.VideoCapture):
            raise TypeError("video_capture must be a cv2.VideoCapture object")
        self.__video_capture = video_capture

    def __parse_frame(self):
        """
        Get then parse the current frame
        """
        if self.__video_capture.isOpened():
            ret, frame = self.__video_capture.read()  # Get the current frame
        if not ret:
            return
        return super().__call__(frame=frame)  # Inference

    def __parse_video_full(self):
        """
        Parse all the video.
        Loop through the frame.
        """
        self.__video_capture.set(2, 0)  # Go to the first frame
        n_frames =  int(self.__video_capture.get(cv2.CAP_PROP_FRAME_COUNT))  # Number of frame
        res = torch.zeros((n_frames, self.get_max_detection(), self.get_points(), 2)) * np.nan  # Instantiate output
        
        for i in tqdm.tqdm(range(n_frames)):
            x = self.__parse_frame()  # Inference
            if not x is None:
                res[i] = x
            else:
                break
        return res

    def __call__(self):
        """
        Parse the full video
        """
        return self.__parse_video_full()
    
    def __iter__(self):
        return self
    
    def __next__(self):
        """
        Return a single frame
        """
        return self.__parse_frame()

    def __open_videocapture(self, source):
        """
        Method to overwrite in subclass
        """
        return source

    def change_source(self, source):
        """
        Change the source
        """
        self.__set_video_capture(video_capture=self.__open_videocapture(source))

class VideoFileParser(VideoCaptureParser):
    """
    Parse a local video
    """
    def __init__(self, video_path, reshape_frame=None, max_detection=100):
        super().__init__(video_capture=self.__open_videocapture(video_path), reshape_frame=reshape_frame, max_detection=max_detection)
    
    def __open_videocapture(self, source):
        """
        Open a cv2.VideoCapture from a path
        """
        if not os.path.exists(source):
            raise FileNotFoundError("{} is not a valid path".format(source))
        return cv2.VideoCapture(source)

class YouTubeParser(VideoCaptureParser):
    """
    Parse a YouTube video from url.
    """
    def __init__(self, video_url, reshape_frame=None, max_detection=100):
        super().__init__(video_capture=self.__open_videocapture(video_url), reshape_frame=reshape_frame, max_detection=max_detection)    

    def __open_videocapture(self, source):
        """
        Open a cv2.VideoCapture from youtube url
        """
        pfy = pafy.new(source)
        play = pfy.getbest(preftype="mp4")
        return cv2.VideoCapture(play.url)

class RpiCamParser(VideoCaptureParser):
    """
    Parse the rpi cam shot
    """
    def __init__(self, max_detection=100):
        super().__init__(video_capture=self.__open_videocapture(), reshape_frame=None, max_detection=max_detection)
    
    def __open_videocapture(self):
        """
        Open cv2.VideoCapture from a int
        """

        camSet = 'nvarguscamerasrc wbmode=3 tnr-mode=2 tnr-strength=1 ee-mode=2 ee-strength=1 !  video/x-raw(memory:NVMM), '
        camSet += 'width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method=2 ! video/x-raw, width=224, '
        camSet += 'height=224, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! videobalance contrast=1.5 brightness=-.2'
        camSet += ' saturation=1.2 !  appsink'
        return cv2.VideoCapture(camSet)
    
    def __call__(self):
        """
        Overwrite call.
        The number of frames is infinite
        """
        return next(self)
