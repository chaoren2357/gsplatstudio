from pathlib import Path
from abc import abstractmethod, ABC
import torch
from gsplatstudio.utils.config import parse_structured


class BaseTrainer(ABC):
    def __init__(self, cfg, logger) -> None:
        self.cfg = parse_structured(self.config_class, cfg)
        self.logger = logger
        self.first_iteration = 1
        self.iteration = 0
    
    @property
    @abstractmethod
    def config_class(self):
        pass

    @property
    @abstractmethod
    def state(self):
        return {}

    @abstractmethod
    def setup_components(self):
        pass

    @abstractmethod
    def restore_components(self,system_path):
        pass

    def init_components(self, recorder, progress_bar, dirs, **kwargs):
        self.recorder = recorder
        self.progress_bar = progress_bar
        for key, value in dirs.items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)


    @abstractmethod
    def train(self):
        pass


    def save_ckpt(self, iteration):
        self.logger.info(f"Saving Checkpoint in ITER {iteration}")
        filename = f"{iteration}.pth"
        torch.save(self.state, str(Path(self.ckpt_dir) / filename))
    
    def save_scene(self, iteration):
        self.logger.info(f"Saving Gaussians in ITER {iteration}")
        ply_path = Path(self.view_dir) / f"point_cloud/iteration_{iteration}" / "point_cloud.ply"
        ply_path.parent.mkdir(parents=True, exist_ok=True)
        self.representation.save_ply(ply_path)
    
    def set(self, name, value):
        setattr(self, name, value)
