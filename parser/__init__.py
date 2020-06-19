import numpy as np
import torch
import torchvision.transforms as transforms
import PIL.Image
import cv2
import torch2trt


class _ImageParser(object):
    """ImageParser object, perform trt_pose, pose estimation to a single frame"""

    def __init__(self, trt_model, parse_objects, points=18):
        """
        :param trt_model: torch2trt model (torch model optimized for tensorRT)
        :param parse_objects: trt_pose.parse_objects.ParseObjects
        """
        self.trt_model = trt_model
        self.parse_objects = parse_objects
        self.points = points

    def _parse_image(self, frame, max_detection=100):
        """
        Parse a single frame
        :param frame: 3 dimensional np.ndarray
        :param max_detection: Maximal number of person detected
        :return: a tensor of shape (1, max_detection, number of points, 2) Chanel 0 : y; Chanel 1: x
        """
        height, width, _ = frame.shape
        npoints = self.points
        
        out = torch.zeros((1, max_detection, npoints, 2)) * np.nan

        mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
        std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
        device = torch.device('cuda')
        data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        data = PIL.Image.fromarray(data)
        data = transforms.functional.to_tensor(data).to(device)
        data.sub_(mean[:, None, None]).div_(std[:, None, None])
        data = data[None, ...]

        cmap, paf = self.trt_model(data)
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        counts, objects, normalized_peaks = self.parse_objects(cmap, paf)
        count = min(int(counts[0]), max_detection)  # number detected objects

        if count > 0:
            objects = objects[0, 0:count]
            npoints = objects[0].shape[0]
            normalized_peaks = normalized_peaks[0, 0:npoints, :count, :]
            normalized_peaks = torch.transpose(normalized_peaks, 0, 1)
            objects = torch.cat([objects.unsqueeze(-1), objects.unsqueeze(-1)], 2)
            normalized_peaks = torch.where(objects >= 0, normalized_peaks, normalized_peaks * np.nan)
            normalized_peaks[:, :, 1] *= width
            normalized_peaks[:, :, 0] *= height

            shp = normalized_peaks.shape
            out[0][:shp[0], :shp[1], :shp[2]] = normalized_peaks

            return out


class _VideoParser(_ImageParser):
    """VideoParser mother class"""

    def __init__(self, trt_model, parse_objects, points=18):
        """
        :param trt_model: torch2trt model (torch model optimized for tensorRT)
        :param parse_objects: trt_pose.parse_objects.ParseObjects
        """
        super().__init__(trt_model, parse_objects, points)

    def _parse_video_stream(self, videocapture, max_detection=100, reshape_frame=None, stream_size=1):
        """
        Create a generator that stream the data
        :param videocapture: a cv2.VideoCapture object
        :param max_detection: limit the number of persons detected
        :param reshape_frame: a ReshapePic1 or ReshapePic2 object or a function that return an array from an array
        :param stream_size: Number of frame returned per batch
        :return: tensor (stream_size, max_detection, points, 2)
        """

        points = self.points
        res = torch.zeros((0, max_detection, points, 2))

        while videocapture.isOpened():
            _, frame = videocapture.read()

            if frame is None:
                break

            if reshape_frame is not None:
                frame = reshape_frame(frame)

            out = self._parse_image(frame, max_detection)  # shape (1, max_detection, points:18, 2)
            res = torch.cat([res, torch.zeros((1, max_detection, points, 2)) * np.nan], dim=0)  # Add one nan tensor

            if out is not None:
                # Add detected values
                res[-1] = out

            if res.shape[0] > stream_size:
                # Limit the tensor to the stream_size last tensors
                res = res[-stream_size:]

            if stream_size >= res.shape[0]:
                yield res
    
    def _parse_video_full(self, videocapture, max_detection=100, reshape_frame=None):
        """
        Parse all the video
        :param videocapture: a cv2.VideoCapture object
        :param max_detection: limit the number of persons detected
        :param reshape_frame: a ReshapePic1 or ReshapePic2 object or a function that return an array from an array
        :return: full tensor (frames, max_detection, points, 2)
        """

        points = self.points
        print(points)
        res = torch.zeros((0, max_detection, points, 2))
        print(res.shape)

        out = self._parse_video_stream(videocapture=videocapture, max_detection=max_detection, reshape_frame=reshape_frame, stream_size=1)
        
        while True:
            #  Loop through the generator
            try:
                z = next(out)
            except:
                z = None
                break

            if z is not None:
                res = torch.cat([res, z], dim=0)

        return res

