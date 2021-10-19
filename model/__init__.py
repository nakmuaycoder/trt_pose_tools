""""
Tools for models

@ load original resnet18 detector with load_original
@ Optimized model with trt_converter
@ load optimized model using load_optz

author: nakmuayoder
date 10/2021
"""

import json
import trt_pose.coco
import torch2trt
import torch
import trt_pose.models
import os
from google_drive_downloader import GoogleDriveDownloader as gdd
from torch2trt import TRTModule
from trt_pose.parse_objects import ParseObjects

__path = os.path.dirname(os.path.abspath(__file__))
__original_model = "resnet18_baseline_att_224x224_A_epoch_249.pth"
__optimized_model = "resnet18_{}_224x224_optz.pth"


def download_model():
    """
    Download pose estimation model from google drive
    """
    gdd.download_file_from_google_drive(file_id='1XYDdCUdiF2xxx4rznmLb62SdOUZuoNbd', dest_path=os.path.join(__path, __original_model))

def load_original():
    """
    Load original model 
    """
    path = os.path.join(__path, __original_model)
    if not os.path.exists(path):
        download_model()
    return torch.load(path)

def trt_converter(batch_size=1):
    """
    Convert torch model to trt
    """
    mdl = os.path.join(__path, __optimized_model.format(batch_size))
    if os.path.exists(mdl):
        return

    with open(os.path.join(__path, "person.json"), 'r') as f:
        human_pose = json.load(f)
    
    num_parts = len(human_pose['keypoints'])
    num_links = len(human_pose['skeleton'])
    print("Dl info")
    model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
    print("Load model")    
    model.load_state_dict(load_original())
    print("Start conversion")
    model_trt = torch2trt.torch2trt(model, [torch.zeros((batch_size, 3, 224, 224)).cuda()], fp16_mode=True, max_workspace_size=1<<25)
    print("Save model")
    torch.save(model_trt.state_dict(), mdl)

def load_optz(batch_size=1):
    """
    Load opptimized model
    """
    path = os.path.join(__path, __optimized_model.format(batch_size))
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(path))
    return model_trt

def  get_parse_objects():
    """
    return a parse object
    """
    path = os.path.join(__path, "person.json")
    with open(path, 'r') as f:
        human_pose = json.load(f)
    topology = trt_pose.coco.coco_category_to_topology(human_pose)
    return ParseObjects(topology)