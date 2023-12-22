import gsplatstudio 
import json
import random
from gsplatstudio.utils.config import parse_structured
from gsplatstudio.utils.type_utils import *
from gsplatstudio.utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from gsplatstudio.data.colmap_helper import *
from dataclasses import dataclass


@dataclass
class ColmapDataModuleConfig:
    source_path: str = '/mnt/d/data/GaussianData/carla_sample'
    eval: bool = False
    data_device: str = 'cuda'
    resolution: int = -1
    images: str = 'images'
    model_path: str = ''
    shuffle: bool = True


@gsplatstudio.register("colmap")
class ColmapDataModule:
    def __init__(self, cfg) -> None:
        self.cfg = parse_structured(ColmapDataModuleConfig, cfg)
        self.resolution_scales = [1.0]
        scene_info = readColmapSceneInfo(self.cfg.source_path, self.cfg.images, self.cfg.eval)
        
        # Save point cloud data
        with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.cfg.model_path, "input.ply") , 'wb') as dest_file:
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
        with open(os.path.join(self.cfg.model_path, "cameras.json"), 'w') as file:
            json.dump(json_cams, file)
        
        # Shuffle the dataset
        if self.cfg.shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling
        
        # TODO: ??What is this
        self.cameras_extent = scene_info.nerf_normalization["radius"]
        self.train_cameras, self.test_cameras = {},{}
        for resolution_scale in self.resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, self.cfg)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, self.cfg)
        self.scene_info = scene_info

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]