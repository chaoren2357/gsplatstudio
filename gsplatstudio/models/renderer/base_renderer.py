from abc import abstractmethod, ABC
from gsplatstudio.utils.type_utils import *
from gsplatstudio.utils.config import parse_structured


class BaseRenderer(ABC):
    def __init__(self, cfg, logger) -> None:
        self.cfg = parse_structured(self.config_class, cfg)
        self.logger = logger

    @property
    @abstractmethod
    def config_class(self):
        pass

    @abstractmethod
    def render(self, representation, camera):
        pass