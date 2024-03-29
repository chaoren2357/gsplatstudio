from abc import abstractmethod, ABC
from gsplatstudio.utils.type_utils import *
from gsplatstudio.utils.config import parse_structured
import torch.nn as nn


class BaseLoss(nn.Module, ABC):
    def __init__(self, cfg, logger) -> None:
        super(BaseLoss, self).__init__()
        self.cfg = parse_structured(self.config_class, cfg)
        self.logger = logger

    @property
    @abstractmethod
    def config_class(self):
        pass

    @abstractmethod
    def forward(self, predict,gt):
        pass


    
