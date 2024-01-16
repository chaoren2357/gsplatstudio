from abc import abstractmethod, ABC
from gsplatstudio.utils.type_utils import *
from gsplatstudio.utils.config import parse_structured


class BaseParamOptim(ABC):
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
    def _restore(self, state, max_iter, **kwargs):
        pass

    @abstractmethod
    def init_optim(self, max_iter, **kwargs):
        pass
        
    @abstractmethod 
    def update_optim(self, iteration):
        pass

    def restore(self, state, max_iter, **kwargs):
        self.logger.info(f"Start restoring paramOptim {self.__class__.__name__}...")
        self._restore(state, max_iter, **kwargs)
        self.logger.info(f"End restoring paramOptim {self.__class__.__name__}...")


