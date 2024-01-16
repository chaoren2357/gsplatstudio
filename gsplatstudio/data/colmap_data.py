import gsplatstudio 
from gsplatstudio.utils.type_utils import *

import json
import random
import shutil
from gsplatstudio.data.colmap_helper import *
from gsplatstudio.data.base_data import BaseDataModule


@dataclass
class ColmapDataModuleConfig:
    processor_type: str
    processor: str
    source_path: str
    eval: int = 0
    device: str = 'cuda'
    resolution: int = -1
    resolution_scales: list = field(default_factory=list)
    shuffle: bool = True


@gsplatstudio.register("colmap-data")
class ColmapDataModule(BaseDataModule):
    
    @property
    def config_class(self):
        return ColmapDataModuleConfig

    def _run(self):
        self.point_cloud, pair_list, ply_path = load_colmap_folder(self.cfg.source_path)

        # Define train and test dataset
        if self.cfg.eval != 0:
            self.train_pair_list = [c for idx, c in enumerate(pair_list) if idx % self.cfg.eval != 0]
            self.test_pair_list = [c for idx, c in enumerate(pair_list) if idx % self.cfg.eval == 0]
        else:
            self.train_pair_list = pair_list
            self.test_pair_list = []

        self.spatial_scale = get_spatial_scale(self.train_pair_list)["radius"]

        input_ply_path = Path(self.view_dir) / "input.ply"
        camera_path = Path(self.view_dir) /  "cameras.json"

        # Save point cloud data
        with open(ply_path, 'rb') as src_file, open(input_ply_path , 'wb') as dest_file:
            dest_file.write(src_file.read())
        # Save camera data
        json_cams = [camera_image_pair.json for camera_image_pair in pair_list]
        with open(camera_path, 'w') as file:
            json.dump(json_cams, file)
        
        # Shuffle the dataset
        if self.cfg.shuffle:
            random.shuffle(self.train_pair_list)  # Multi-res consistent random shuffling
            random.shuffle(self.train_pair_list)  # Multi-res consistent random shuffling

    def _restore(self, data_source_path, iteration):

        data_source_path = Path(data_source_path)
        data_target_path = Path(self.view_dir)

        # Copy files for viewers
        for file_name in ["camera.json", "cfg_args", "input.ply"]:
            source_file = data_source_path / file_name
            target_file = data_target_path / file_name
            if source_file.exists():
                shutil.copy(source_file, target_file)

        source_folder = data_source_path / "point_cloud" / f"iteration_{iteration}"
        target_folder = data_target_path / "point_cloud" / f"iteration_{iteration}"
        if source_folder.exists():
            shutil.copytree(source_folder, target_folder)

        self.point_cloud, pair_list, ply_path = load_colmap_folder(self.cfg.source_path)

        # Define train and test dataset
        if self.cfg.eval != 0:
            self.train_pair_list = [c for idx, c in enumerate(pair_list) if idx % self.cfg.eval != 0]
            self.test_pair_list = [c for idx, c in enumerate(pair_list) if idx % self.cfg.eval == 0]
        else:
            self.train_pair_list = pair_list
            self.test_pair_list = []

        self.spatial_scale = get_spatial_scale(self.train_pair_list)["radius"]

        input_ply_path = Path(self.view_dir) / "input.ply"
        camera_path = Path(self.view_dir) /  "cameras.json"

        # Save point cloud data
        with open(ply_path, 'rb') as src_file, open(input_ply_path , 'wb') as dest_file:
            dest_file.write(src_file.read())
        # Save camera data
        json_cams = [camera_image_pair.json for camera_image_pair in pair_list]
        with open(camera_path, 'w') as file:
            json.dump(json_cams, file)
        
        # Shuffle the dataset
        if self.cfg.shuffle:
            random.shuffle(self.train_pair_list)  # Multi-res consistent random shuffling
            random.shuffle(self.train_pair_list)  # Multi-res consistent random shuffling

    def get_train_pair_list(self):
        return self.train_pair_list

    def get_test_pair_list(self):
        return self.test_pair_list