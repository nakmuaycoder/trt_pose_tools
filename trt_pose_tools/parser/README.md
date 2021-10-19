# Parser

This package contains differents kind of the parser for Image and Video.

|Type|Type of input|Parser|
|:----:|:----:|:----:|
|Image|```numpy.ndarray```|[ImageParser](#Image-Parser)|
|Video|```cv2.VideoCapture```|[VideoCaptureParser](#VideoCaptureParser)|
|Video|File|[VideoFileParser](#VideoFileParser)|
|Video|YouTube video|[YouTubeParser](#YouTubeParser)|
|Video| Raspberry Pi Cam| [RpiCamParser](#RpiCamParser)
|Video| Custom parser| [Create your own parser](#Create-customs-VideoParsers)

## [Image Parser](image_parser.py)

Parse a single image.
Return a tensor of shape (max_detection, 18, 2)

```py
import cv2
from trt_pose_tools.parser import ImageParser
from trt_pose_tools.image_preprocessing.frame_reshaping import ReshapePic1

image = cv2.imread("img_doc/test.jpg")  # Read the image
parser = ImageParser(max_detection=100, reshape_fram=ReshapePic1(shape=(224,224,3)))  # Parser instantiation
parser(image)  # Inference
```

## [video_parser](video_parser.py)

Parse videos using different objects.

### VideoCaptureParser

```py
from trt_pose_tools.parser import VideoCaptureParser
from trt_pose_tools.image_preprocessing.frame_reshaping import ReshapePic1

video_capture1 = cv2.VideoCapture("path_to_my_video1")  # Open a VideoCapture
video_capture2 = cv2.VideoCapture("path_to_my_video2")  # Open a VideoCapture

parser = VideoCaptureParser(video_capture=video_capture1, reshape_frame=ReshapePic1((224, 224)), max_detection=2)  # Parser instantiation
next(parser)  # Parse the current frame
parser()  # Parse the whole video

parser.change_source(video_capture2)  # Change the source
```
### VideoFileParser

```py
from trt_pose_tools.parser import VideoFileParser
from trt_pose_tools.image_preprocessing.frame_reshaping.frame_reshaping import ReshapePic1

video_path = "path_to_my_video"
parser = VideoFileParser(video_path=video_path, reshape_frame=ReshapePic1((224, 224)), max_detection=2)
```

### YouTubeParser

```py
from trt_pose_tools.parser import YouTubeParser
from trt_pose_tools.image_preprocessing.frame_reshaping import ReshapePic1

video_url = "https://www.youtube.com/watch?v=0z3fw2yXC5I"
parser = YouTubeParser(video_path=video_path, reshape_frame=ReshapePic1((224, 224)), max_detection=2)
```
### RpiCamParser

```py
from trt_pose_tools.parser import RpiCamParser

parser = YouTubeParser(max_detection=2)
```

The ```__call__``` and ```__next__``` RpiCamParser method parse a single frame.
The frame shape of the recorded frame is (224, 224, 3).
User dont have to reshape it before the inference.

### Create customs VideoParsers

User can create custom VideoParser based on a ```VideoCaptureParser```:

```py
from trt_pose_tools.parser import VideoCaptureParser

class myCustomParser(VideoCaptureParser):
    def __init__(self, source, reshape_frame=None, max_detection=100):
        super().__init__(video_capture=self.__open_videocapture(source), reshape_frame=reshape_frame, max_detection=max_detection)
    
    def __open_videocapture(self, source):
        """
        Open a videocapture here
        """
        pass
```
