import gsplatstudio
import collections
import numpy as np
from pathlib import Path
from gsplatstudio.utils.graphics_utils import qvec2rotmat

from abc import ABC, abstractmethod
from gsplatstudio.utils.config import parse_structured

class GCameraInfo:
    def __init__(self,uid: int,
                    R: np.array,
                    T: np.array,
                    FovY: np.array,
                    FovX: np.array,
                    image: np.array,
                    image_path: str,
                    image_name: str,
                    width: int,
                    height: int):
        self.uid = uid
        self.R = R
        self.T = T
        self.FovY = FovY
        self.FovX = FovX
        self.image = image
        self.image_path = image_path
        self.image_name = image_name
        self.width = width
        self.height = height

class GCamera:
    def __init__(self, uid: int,
                       model: str, 
                       width: int, 
                       height: int, 
                       params: np.array):
        self.uid = uid
        self.model = model
        self.params = params
        self.width = width
        self.height = height
    
class GImage:
    def __init__(self,uid: int,
                    qvec: np.array,
                    tvec: np.array,
                    xys: np.array,
                    point3D_ids: np.array,
                    camera_id: int,
                    name: str):
        self.uid = uid
        self.camera_id = camera_id
        self.name = name
        self.qvec = qvec
        self.tvec = tvec
        self.point3D_ids = point3D_ids
        self.xys = xys

    @property
    def R(self):
        return np.transpose(qvec2rotmat(self.qvec))
    @property
    def T(self):
        return np.array(self.tvec)

class GPointCloud:
    def __init__(self, points: np.array, 
                       color: np.array,
                       normals: np.array,
                       **kwargs):
        self.points = points
        self.color = color
        self.normals = normals
        for key, value in kwargs.items():
            setattr(self, key, value)

class GDataset:
    def __init__(self, point_cloud: GPointCloud, 
                       train_cameras: list,
                       test_cameras: list,
                       spatial_scale: dict,
                       ply_path: str,
                       **kwargs):
        self.point_cloud = point_cloud
        self.train_cameras = train_cameras
        self.test_cameras = test_cameras
        self.spatial_scale = spatial_scale
        self.ply_path = Path(ply_path)
        for key, value in kwargs.items():
            setattr(self, key, value)

GCameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])

CAMERA_MODELS = {
    GCameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    GCameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    GCameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    GCameraModel(model_id=3, model_name="RADIAL", num_params=5),
    GCameraModel(model_id=4, model_name="OPENCV", num_params=8),
    GCameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    GCameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    GCameraModel(model_id=7, model_name="FOV", num_params=5),
    GCameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    GCameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    GCameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}

GCAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])

class BaseDataModule(ABC):
    def __init__(self, cfg, logger, trial_dir) -> None:
        self.cfg = parse_structured(self.config_class, cfg)
        self.logger = logger
        self.trial_dir = trial_dir
        self.data_processor = gsplatstudio.find(self.cfg.processor_type)(self.cfg.processor, self.logger, self.cfg.source_path)
    
    @property
    @abstractmethod
    def config_class(self):
        pass

    def run(self):
        if not self.data_processor.should_skip:
            self.data_processor.run()
        else:
            self.logger.info("Skip preprocessing data...")
        