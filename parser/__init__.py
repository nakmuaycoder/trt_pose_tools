import numpy as np
import torch
import torchvision.transforms as transforms
import PIL.Image
import cv2


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
        self.max_batch_size = trt_model.engine.max_batch_size

    def _preprocess(self, single_frame):
        mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
        std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
        device = torch.device('cuda')
        data = cv2.cvtColor(single_frame, cv2.COLOR_BGR2RGB)
        data = PIL.Image.fromarray(data)
        data = transforms.functional.to_tensor(data).to(device)
        data.sub_(mean[:, None, None]).div_(std[:, None, None])
        return data[None, ...]

    def _parse_image(self, frame, max_detection=100):
        """
        Parse a single frame
        :param frame: 3 dimensional np.ndarray or list of np.ndarray
        :param max_detection: Maximal number of person detected
        :return: a tensor of shape (1, max_detection, number of points, 2) Chanel 0 : y; Chanel 1: x
        """

        height, width, _ = frame[0].shape
        data = torch.cat([self._preprocess(single_frame=_) for _ in frame])

        cmap, paf = self.trt_model(data)
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        counts, objects, normalized_peaks = self.parse_objects(cmap, paf)

        batch_size, _, point = objects.shape
        out = torch.zeros((batch_size, max_detection, point, 2)) * np.nan

        for frame in range(batch_size):
            for obj_index in range(min(counts[frame], max_detection)):
                obj = objects[frame, obj_index]
                out[frame, obj_index] = torch.Tensor([[normalized_peaks[frame, i, _, 0], normalized_peaks[frame, i, _, 1]] if _ >= 0 else [np.nan, np.nan] for i, _ in enumerate(obj)])
        out[:, :, :, 0] *= width
        out[:, :, :, 1] *= height
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
        :return: gennerator tensor (stream_size, max_detection, points, 2)
        """

        points = self.points
        batch_size = self.max_batch_size
        res = torch.zeros((0, max_detection, points, 2))
        batch_output = []

        while videocapture.get(1) <= stream_size and videocapture.get(1) <= videocapture.get(cv2.CAP_PROP_FRAME_COUNT):
            # Initialisation, get the first stream size values
            batch = [videocapture.read()[1] for _ in range(batch_size)]

            if not reshape_frame is None:
                batch = [reshape_frame(_) for _ in batch if not _ is None]
            
            batch_output += batch
            z = self._parse_image(frame=batch, max_detection=max_detection)
            res = torch.cat([res, z], dim=0)

        yield res[:stream_size], batch_output[:stream_size]  # First output

        while videocapture.get(1) <= videocapture.get(cv2.CAP_PROP_FRAME_COUNT):
            batch_output = batch_output[1:]
            batch = [videocapture.read()[1] for _ in range(batch_size) if not _ is None]

            if not reshape_frame is None:
                batch = [reshape_frame(_) for _ in batch if not _ is None]

            batch_output += batch
            z = self._parse_image(frame=batch, max_detection=max_detection)
            res = torch.cat([res[1:], z], dim=0)

            yield res[:stream_size], batch_output[:stream_size]

            while res.shape[0] >= stream_size:
                res = res[1:]
                batch_output = batch_output[1:]
                yield res[:stream_size]

    def _parse_video_full(self, videocapture, max_detection=100, reshape_frame=None):
        """
        Parse all the video
        :param videocapture: a cv2.VideoCapture object
        :param max_detection: limit the number of persons detected
        :param reshape_frame: a ReshapePic1 or ReshapePic2 object or a function that return an array from an array
        :return: full tensor (frames, max_detection, points, 2)
        """

        points = self.points
        batch_size = self.max_batch_size
        res = torch.zeros((0, max_detection, points, 2))

        while videocapture.get(1) <= videocapture.get(cv2.CAP_PROP_FRAME_COUNT) - batch_size:
            batch = [videocapture.read()[1] for _ in range(batch_size)]

            if not reshape_frame is None:
                batch = [reshape_frame(_) for _ in batch if not _ is None]

            z = self._parse_image(frame=batch, max_detection=max_detection)
            res = torch.cat([res, z], dim=0)

        while videocapture.get(1) < videocapture.get(cv2.CAP_PROP_FRAME_COUNT):
            batch += [videocapture.read()[1]]

        return res
