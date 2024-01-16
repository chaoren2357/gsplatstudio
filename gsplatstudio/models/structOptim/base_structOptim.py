from abc import abstractmethod, ABC
from gsplatstudio.utils.type_utils import *
from gsplatstudio.utils.config import parse_structured


class BaseStructOptim(ABC):
    def __init__(self, cfg, logger) -> None:
        self.cfg = parse_structured(self.config_class, cfg)
        self.logger = logger

    @property
    @abstractmethod
    def config_class(self):
        pass

    @property
    @abstractmethod
    def state(self):
        pass

    @abstractmethod
    def _restore(self, state, **kwargs):
        pass

    @abstractmethod
    def init_optim(self, **kwargs):
        pass
        
    @abstractmethod 
    def update_optim(self, iteration, model, paramOptim, render_pkg, **kwargs):
        pass

    def restore(self, state, **kwargs):
        self.logger.info(f"Start restoring structOptim {self.__class__.__name__}...")
        self._restore(state, **kwargs)
        self.logger.info(f"End restoring structOptim {self.__class__.__name__}...")

