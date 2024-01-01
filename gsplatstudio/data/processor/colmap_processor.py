
import gsplatstudio
from gsplatstudio.utils.type_utils import *
from gsplatstudio.data.processor.base_processor import BaseDataProcessor
import subprocess
from pathlib import Path



@dataclass
class ColmapProcessorConfig:
    use_gpu: bool = True
    skip_matching: bool = False
    camera: str = "OPENCV"
    map_ba_global_function_tolerance: float = 0.000001

@gsplatstudio.register("colmap-processor")
class ColmapProcessor(BaseDataProcessor):
    def __init__(self, cfg, logger, source_path) -> None:
        super().__init__(cfg, logger, source_path)
    
    @property
    def config_class(self):
        return ColmapProcessorConfig
    
    def run(self):
        self.logger.info("Start running ColmapProcessor...")
        if not self.cfg.skip_matching:
            sparse_path = Path(self.source_path_str) / "distorted" / "sparse"
            sparse_path.mkdir(parents=True, exist_ok=True)
            
            ## Feature extraction
            feature_extractor_cmd = "colmap feature_extractor" + \
                      f" --database_path {self.source_path_str}/distorted/database.db" + \
                      f" --image_path {self.source_path_str}/input" + \
                      f" --ImageReader.single_camera 1" + \
                      f" --ImageReader.camera_model {self.cfg.camera}" + \
                      f" --SiftExtraction.use_gpu {int(self.cfg.use_gpu)}"
            
            result = subprocess.run(feature_extractor_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            exit_code = result.returncode
            if exit_code != 0:
                self.logger.error(f"Feature extraction failed with code {exit_code}. Exiting.")
                exit(exit_code)
            self.logger.info("Finish feature extraction...")
            ## Feature matching
            feat_matching_cmd = "colmap exhaustive_matcher" + \
                      f" --database_path {self.source_path_str}/distorted/database.db" + \
                      f" --SiftMatching.use_gpu {int(self.cfg.use_gpu)}"
            result = subprocess.run(feat_matching_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            exit_code = result.returncode
            if exit_code != 0:
                self.logger.error(f"Feature matching failed with code {exit_code}. Exiting.")
                exit(exit_code)
            self.logger.info("Finish feature matching...")
            ## Bundle adjustment
            # The default Mapper tolerance is unnecessarily large, decreasing it speeds up bundle adjustment steps.
            mapper_cmd = "colmap mapper" + \
                      f" --database_path {self.source_path_str}/distorted/database.db" + \
                      f" --image_path {self.source_path_str}/input" + \
                      f" --output_path {self.source_path_str}/distorted/sparse" + \
                      f" --Mapper.ba_global_function_tolerance={self.cfg.map_ba_global_function_tolerance}"
            result = subprocess.run(mapper_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            exit_code = result.returncode
            if exit_code != 0:
                self.logger.error(f"Mapper failed with code {exit_code}. Exiting.")
                exit(exit_code)
            self.logger.info("Finish mapping...")
        ### Image undistortion(undistort our images into ideal pinhole intrinsics)
        img_undist_cmd = "colmap image_undistorter" + \
                      f" --image_path {self.source_path_str}/input" + \
                      f" --input_path {self.source_path_str}/distorted/sparse/0" + \
                      f" --output_path {self.source_path_str}" + \
                      f" --output_type COLMAP"
        result = subprocess.run(img_undist_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        exit_code = result.returncode
        if exit_code != 0:
            self.logger.error(f"Mapper failed with code {exit_code}. Exiting.")
            exit(exit_code)
        self.logger.info("Finish image undistorter...")
        
        sparse_result_path = Path(self.source_path_str) / "sparse" 
        destination_path = sparse_result_path / "0"
        destination_path.mkdir(exist_ok=True)

        for file in sparse_result_path.iterdir():
            if file.name == '0':
                continue
            source_file = file
            destination_file = destination_path / file.name
            source_file.rename(destination_file)

        self.logger.info(f"Finish processing data, source path: {self.source_path_str}")


