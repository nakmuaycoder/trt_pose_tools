# trt_pose_tools

This programm is made for run on jetson nano and contain additional tools for nvidia [trt_pose](https://github.com/NVIDIA-AI-IOT/trt_pose/tree/master/trt_pose) library.

Install's instructions are available in the [README.md file](https://github.com/NVIDIA-AI-IOT/trt_pose/blob/master/README.md) of the trt_pose project.

## trt_pose_tools' content

### parser package

This package contain the parser for Image and Video.
First load the model and the ParseObject.

```python
import cv2
import trt_pose.coco
from trt_pose.parse_objects import ParseObjects
import torch2trt
import json
from parser.image_parser import ReshapePic1

#Load the model
mdl = "resnet18_baseline_att_224x224_A_epoch_249_trt.pth"
model_trt = torch2trt.TRTModule()
model_trt.load_state_dict(torch.load(mdl))

# Instantiate a ParsObjects
topology = "trt_pose/tasks/human_pose/human_pose.json"
with open(topology, 'r') as f:
    human_pose = json.load(f)
topology = trt_pose.coco.coco_category_to_topology(human_pose)
parse_object = ParseObjects(topology)

rsp = ReshapePic1(shape=(224,224,3))
```

- Image parser
```python
from parser.image_parser import ImageParser
# Capture a single frame using the web cam
cap = cv2.VideoCapture(0)
_, frame = cap.read()
cap.release()

# Reshape the frame
frame = rsp(frame)

# Parse frame
img_parser = ImageParser(trt_model=model_trt, parse_objects=parse_objects)
result = image_parser(frame=frame, max_detection=100)
```

- Video parser
```python
from parser.video_parser import VideoParser, YouTube_VideoParser
url = "https://www.youtube.com/watch?v=0z3fw2yXC5I"

vdo_parser = VideoParser(trt_model=model_trt, parse_objects=parse_objects)
yt_parser = YouTube_VideoParser(trt_model=model_trt, parse_objects=parse_objects)

# Parse a YouTube video
tensor = yt_parser(video_url=url, max_detection=100, reshape_frame=rsp)

# Create a generator using the webcam returning the detected keypoints on the 5 last frame
stream = vdo_parser.stream(video_path=0, max_detection=100, stream_size=5, reshape_frame=rsp)

# Iterate on stream
while True:
    try:
        z = next(stream)
    except:
        z = None
        break
```


## Package Tracking


