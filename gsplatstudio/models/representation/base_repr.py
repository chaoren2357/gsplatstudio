from abc import ABC, abstractmethod
from gsplatstudio.utils.config import parse_structured


class BaseRepr(ABC):
    def __init__(self, cfg, logger) -> None:
        self.cfg = parse_structured(self.config_class, cfg)
        self.logger = logger

    @property
    @abstractmethod
    def config_class(self):
        pass

    @abstractmethod
    def _restore(self, state, **kwargs):
        pass


    def restore(self, state, **kwargs):
        self.logger.info(f"Start restoring representation {self.__class__.__name__}...")
        self._restore(state, **kwargs)
        self.logger.info(f"End restoring representation {self.__class__.__name__}...")

