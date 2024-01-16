
import gsplatstudio
from gsplatstudio.utils.type_utils import *
from gsplatstudio.data.processor.base_processor import BaseDataProcessor
from gsplatstudio.utils.system_utils import load_json
from gsplatstudio.utils.camera_utils import transform_camera_from_carla_matrix_to_colmap_W2C_quaternion, transform_camera_from_matrixcity_matrix_to_colmap_W2C_quaternion
from gsplatstudio.utils.graphics_utils import fov2focal
import sqlite3
from pathlib import Path
import numpy as np


@dataclass
class ColmapWithCamProcessorConfig:
    use_gpu: bool = True
    camera: str = "OPENCV"
    map_ba_global_function_tolerance: float = 0.000001

@gsplatstudio.register("colmap_with_cam-processor")
class ColmapWithCamProcessor(BaseDataProcessor):
    def __init__(self, cfg, logger, source_path) -> None:
        super().__init__(cfg, logger, source_path)
    
    @property
    def config_class(self):
        return ColmapWithCamProcessorConfig
    
    @property
    def should_skip(self):
        cameras_file = Path(self.source_path_str) / "sparse" / "0" / "cameras.bin"
        images_file = Path(self.source_path_str) / "sparse" / "0" / "images.bin"
        points3D_file = Path(self.source_path_str) / "sparse" / "0" / "points3D.bin"
        return cameras_file.exists() and images_file.exists() and points3D_file.exists()
    
    def _run(self):
        project_folder = Path(self.source_path_str) / "distorted"
        project_folder.mkdir(parents=True, exist_ok=True)
        database_path = Path(self.source_path_str) / "distorted" / "database.db"
        image_distorted_folder = Path(self.source_path_str) / "input"
        camera_folder = Path(self.source_path_str) / "camera"

        ## Feature extraction
        feature_extractor_cmd = "colmap feature_extractor" + \
                    f" --database_path {str(database_path)}" + \
                    f" --image_path {str(image_distorted_folder)}" + \
                    f" --ImageReader.single_camera 1" + \
                    f" --ImageReader.camera_model {self.cfg.camera}" + \
                    f" --SiftExtraction.use_gpu {int(self.cfg.use_gpu)}"
        
        exit_code = self.run_command_with_realtime_output(feature_extractor_cmd)
        if exit_code != 0:
            self.logger.error(f"Feature extraction failed with code {exit_code}. Exiting.")
            exit(exit_code)
        self.logger.info("Finish feature extraction...")

        ## Create points3D.txt
        points3D_txt_path = project_folder / 'points3D.txt'
        open(str(points3D_txt_path), 'w').close()

        ## Create camera.txt
        camera_txt_path = project_folder / 'cameras.txt'
        open(str(camera_txt_path), 'w').close()

        unique_cameras = {}
        camera_id = 1
        for camera_file in camera_folder.glob('*.json'):
            camera_data = load_json(camera_file)
            intrinsics = camera_data['intrinsics']
            fov_radians = np.radians(intrinsics['fov'])
            focal_length = fov2focal(fov_radians, intrinsics['width'])
            key = (intrinsics['width'], intrinsics['height'], focal_length)
            if key not in unique_cameras:
                unique_cameras[key] = camera_id
                camera_id += 1
        with open(str(camera_txt_path), 'w') as file:
            for (width, height, focal_length), idx in unique_cameras.items():
                file.write(f"{idx} SIMPLE_PINHOLE {width} {height} {focal_length} {width/2} {height/2}\n")

        ## Create images.txt
        images_txt_path = project_folder / 'images.txt'
        open(str(images_txt_path), 'w').close()
        image_files_with_id = [(int(image_file.stem.split('_')[1]), image_file) for image_file in image_distorted_folder.glob('*.png')]
        sorted_image_files = sorted(image_files_with_id, key=lambda x: x[0])
        with open(str(images_txt_path), 'w') as file:
            for _ , image_file in sorted_image_files:
                image_name = image_file.stem
                image_id = image_name.split('_')[1]
                camera_name = image_name[:-5] + "camera"
                camera_path = str(camera_folder / camera_name) + ".json"
                camera_data = load_json(camera_path)
                quat,trans = transform_camera_from_matrixcity_matrix_to_colmap_W2C_quaternion(camera_data)
                file.write(f"{int(image_id)} {' '.join(map(str, quat))} {' '.join(map(str, trans))} 1 {image_file.name}\n")
                file.write("\n")

        ## modify DB
        conn = sqlite3.connect(str(database_path))
        cursor = conn.cursor()
        for image_file in image_distorted_folder.glob('*.png'):
            image_name = image_file.stem
            image_id = image_name.split('_')[1]
            camera_name = image_name[:-5] + "camera"
            camera_path = str(camera_folder / camera_name) + ".json"
            camera_data = load_json(camera_path)
            quat,trans = transform_camera_from_matrixcity_matrix_to_colmap_W2C_quaternion(camera_data)
            cursor.execute("""
                UPDATE images
                SET prior_qw = ?, prior_qx = ?, prior_qy = ?, prior_qz = ?, prior_tx = ?, prior_ty = ?, prior_tz = ?
                WHERE name = ?
            """, (*quat, *trans, str(image_file.name)))
        conn.commit()
        conn.close()
        
        ## Feature matching
        feat_matching_cmd = "colmap exhaustive_matcher" + \
                    f" --database_path {str(database_path)}" + \
                    f" --SiftMatching.use_gpu {int(self.cfg.use_gpu)}"
        exit_code = self.run_command_with_realtime_output(feat_matching_cmd)
        if exit_code != 0:
            self.logger.error(f"Feature matching failed with code {exit_code}. Exiting.")
            exit(exit_code)
        self.logger.info("Finish feature matching...")
        
        ## Point Triangulator
        sparse_folder = Path(self.source_path_str) / "sparse" / "0"
        sparse_folder.mkdir(parents=True, exist_ok=True)
        point_triangulator_cmd = "colmap point_triangulator" + \
                    f" --database_path {str(database_path)}" + \
                    f" --image_path {str(image_distorted_folder)}" + \
                    f" --input_path {str(project_folder)}" + \
                    f" --output_path {str(sparse_folder)}"
        exit_code = self.run_command_with_realtime_output(point_triangulator_cmd)
        if exit_code != 0:
            self.logger.error(f"Point triangulator failed with code {exit_code}. Exiting.")
            exit(exit_code)
        self.logger.info("Finish point triangulator...")

        ## Bundle adjustment
        # The default Mapper tolerance is unnecessarily large, decreasing it speeds up bundle adjustment steps.
        bundle_adjuster_cmd = "colmap bundle_adjuster" + \
                    f" --input_path {str(sparse_folder)}" + \
                    f" --output_path {str(sparse_folder)}"
        exit_code = self.run_command_with_realtime_output(bundle_adjuster_cmd)
        if exit_code != 0:
            self.logger.error(f"Bundle adjuster failed with code {exit_code}. Exiting.")
            exit(exit_code)
        self.logger.info("Finish bundle adjuster...")
        
        ## Image undistortion(undistort our images into ideal pinhole intrinsics)
        img_undist_cmd = "colmap image_undistorter" + \
                      f" --image_path {str(image_distorted_folder)}" + \
                      f" --input_path {str(sparse_folder)}" + \
                      f" --output_path {self.source_path_str}" + \
                      f" --output_type COLMAP"
        exit_code = self.run_command_with_realtime_output(img_undist_cmd)
        if exit_code != 0:
            self.logger.error(f"Image undistortion failed with code {exit_code}. Exiting.")
            exit(exit_code)
        self.logger.info("Finish image undistorter...")
    



