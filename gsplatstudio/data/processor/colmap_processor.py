import gsplatstudio
from gsplatstudio.utils.type_utils import *
from gsplatstudio.data.processor.base_processor import BaseDataProcessor
from pathlib import Path



@dataclass
class ColmapProcessorConfig:
    use_gpu: bool = True
    camera: str = "OPENCV"
    map_ba_global_function_tolerance: float = 0.000001

@gsplatstudio.register("colmap-processor")
class ColmapProcessor(BaseDataProcessor):
    def __init__(self, cfg, logger, source_path) -> None:
        super().__init__(cfg, logger, source_path)
    
    @property
    def config_class(self):
        return ColmapProcessorConfig
    
    @property
    def should_skip(self):
        cameras_file = Path(self.source_path_str) / "sparse" / "0" / "cameras.bin"
        images_file = Path(self.source_path_str) / "sparse" / "0" / "images.bin"
        points3D_file = Path(self.source_path_str) / "sparse" / "0" / "points3D.bin"
        return cameras_file.exists() and images_file.exists() and points3D_file.exists()
            
    def _run(self):
        sparse_path = Path(self.source_path_str) / "distorted" / "sparse"
        sparse_path.mkdir(parents=True, exist_ok=True)
        
        ## Feature extraction
        feature_extractor_cmd = "colmap feature_extractor" + \
                    f" --database_path {self.source_path_str}/distorted/database.db" + \
                    f" --image_path {self.source_path_str}/input" + \
                    f" --ImageReader.single_camera 1" + \
                    f" --ImageReader.camera_model {self.cfg.camera}" + \
                    f" --SiftExtraction.use_gpu {int(self.cfg.use_gpu)}"
        
        exit_code = self.run_command_with_realtime_output(feature_extractor_cmd)
        if exit_code != 0:
            self.logger.error(f"Feature extraction failed with code {exit_code}. Exiting.")
            exit(exit_code)
        self.logger.info("Finish feature extraction...")
        ## Feature matching
        feat_matching_cmd = "colmap exhaustive_matcher" + \
                    f" --database_path {self.source_path_str}/distorted/database.db" + \
                    f" --SiftMatching.use_gpu {int(self.cfg.use_gpu)}"
        exit_code = self.run_command_with_realtime_output(feat_matching_cmd)
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
        exit_code = self.run_command_with_realtime_output(mapper_cmd)
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
        exit_code = self.run_command_with_realtime_output(img_undist_cmd)
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
    




