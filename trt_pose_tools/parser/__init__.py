"""
parser package
Contains wrapper, performing pose estimation on both image and video.

author : nakmuaycoder
date : 10/2021
"""


from .image_parser import ImageParser
from .video_parser import VideoCaptureParser, VideoFileParser, YouTubeParser, RpiCamParser