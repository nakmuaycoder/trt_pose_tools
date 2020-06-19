"""
VideoParser.
Run trt_pose on videos

Parser for video file and for YouTube video
All of those object inherit from parser._VideoParser class

"""

import os
import cv2
import pafy
from parser import _VideoParser


class VideoParser(_VideoParser):
    """Parse a video from a local file"""

    def __init__(self, trt_model, parse_objects, points=18):
        """
        :param trt_model:
        :param points: number of detected points
        :param parse_objects:
        """
        super().__init__(trt_model, parse_objects, points)

    def __call__(self, video_path, max_detection=100, reshape_frame=None):
        """
        Parse all the video
        :param video_path: path of the video or webcam access (0 or 1)
        :param max_detection: limit the number of persons detected
        :param reshape_frame: a ReshapePic1 or ReshapePic2 object or a function that return an array from an array
        :return: full tensor (frames, max_detection, points, 2)
        """
        videocapture = cv2.VideoCapture(video_path)
        out = self._parse_video_full(videocapture, max_detection=100, reshape_frame=None)
        return out

    def stream(self, video_path, max_detection, stream_size, reshape_frame=None):
        """
        Create a generator that stream the data
        :param video_path: path of the video
        :param max_detection: limit the number of persons detected
        :param reshape_frame: a ReshapePic1 or ReshapePic2 object or a function that return an array from an array
        :param stream_size: Number of frame returned per batch
        :return: tensor (stream_size, max_detection, points, 2)
        """
        if os.path.exists(video_path):
            videocapture = cv2.VideoCapture(video_path)

            out = self._parse_video_stream(videocapture=videocapture, max_detection=max_detection,
                                           reshape_frame=reshape_frame, stream_size=stream_size)
            return out


class YouTube_VideoParser(_VideoParser):
    """A VideoParser for YouTube video stream"""

    def __init__(self, trt_model, parse_objects, points=18):
        """
        :param trt_model:
        :param parse_objects:
        :param points:
        """
        super().__init__(trt_model, parse_objects, points)

    def __call__(self, video_url, max_detection=100, reshape_frame=None):
        """
        Parse all the video
        :param video_url: path of the video
        :param max_detection: limit the number of persons detected
        :param reshape_frame: a ReshapePic1 or ReshapePic2 object or a function that return an array from an array
        :return: full tensor (frames, max_detection, points, 2)
        """
        pfy = pafy.new(video_url)
        play = pfy.getbest(preftype="mp4")
        videocapture = cv2.VideoCapture(play.url)
        out = self._parse_video_full(videocapture, max_detection, reshape_frame)
        videocapture.release()
        return out

    def stream(self, video_url, max_detection, stream_size, reshape_frame=None):
        """
        Create a generator that stream the data
        :param video_url: url of the video
        :param max_detection: limit the number of persons detected
        :param reshape_frame: a ReshapePic1 or ReshapePic2 object or a function that return an array from an array
        :param stream_size: Number of frame returned per batch
        :return: tensor (stream_size, max_detection, points, 2)
        """
        pfy = pafy.new(video_url)
        play = pfy.getbest(preftype="mp4")
        videocapture = cv2.VideoCapture(play.url)
        out = self._parse_video_stream(videocapture=videocapture, max_detection=max_detection,
                                       reshape_frame=reshape_frame, stream_size=stream_size)
        videocapture.release()
        return out

