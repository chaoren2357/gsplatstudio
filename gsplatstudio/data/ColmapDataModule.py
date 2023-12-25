import gsplatstudio 
import json
import random
from gsplatstudio.utils.config import parse_structured
from gsplatstudio.utils.type_utils import *
from gsplatstudio.utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from gsplatstudio.data.colmap_helper import *


@dataclass
class ColmapDataModuleConfig:
    processor_type: str
    processor: str
    source_path: str = '/mnt/d/data/GaussianData/carla_sample'
    eval: bool = False
    data_device: str = 'cuda'
    resolution: int = -1
    resolution_scales: list = field(default_factory=list)
    shuffle: bool = True


@gsplatstudio.register("colmap-data")
class ColmapDataModule:
    def __init__(self, cfg, trial_dir) -> None:
        self.trial_dir = trial_dir
        self.cfg = parse_structured(ColmapDataModuleConfig, cfg)
        scene_info = load_colmap_folder(self.cfg.source_path, self.cfg.eval)
        input_ply_path = Path(self.trial_dir) / "input.ply"
        camera_path = Path(self.trial_dir) /  "cameras.json"
        # Save point cloud data
        with open(scene_info.ply_path, 'rb') as src_file, open(input_ply_path , 'wb') as dest_file:
            dest_file.write(src_file.read())
        
        # Save camera data
        json_cams = []
        camlist = []
        if scene_info.test_cameras:
            camlist.extend(scene_info.test_cameras)
        if scene_info.train_cameras:
            camlist.extend(scene_info.train_cameras)
        for id, cam in enumerate(camlist):
            json_cams.append(camera_to_JSON(id, cam))
        with open(camera_path, 'w') as file:
            json.dump(json_cams, file)
        
        # Shuffle the dataset
        if self.cfg.shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling
        
        self.spatial_scale = scene_info.spatial_scale["radius"]
        self.train_cameras, self.test_cameras = {},{}
        for resolution_scale in self.cfg.resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, self.cfg)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, self.cfg)
        self.scene_info = scene_info

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]