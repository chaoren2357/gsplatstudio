from abc import ABC, abstractmethod

import gsplatstudio
from gsplatstudio.utils.config import parse_structured


class BaseDataModule(ABC):
    def __init__(self, cfg, logger, dirs) -> None:
        self.cfg = parse_structured(self.config_class, cfg)
        self.logger = logger
        for key, value in dirs.items():
            setattr(self, key, value)
        self.data_processor = gsplatstudio.find(self.cfg.processor_type)(self.cfg.processor, self.logger, self.cfg.source_path)
    
    @property
    @abstractmethod
    def config_class(self):
        pass

    @abstractmethod
    def _restore(self, data_path, iteration):
        pass

    @abstractmethod
    def _run(self):
        pass

    def restore(self, data_path, iteration):
        self.logger.info(f"Start restoring data module {self.__class__.__name__} from {data_path} of iteration {iteration}...")
        self._restore(data_path, iteration)
        self.logger.info(f"End restoring data module {self.__class__.__name__}...")

    def run(self):
        if not self.data_processor.should_skip:
            self.data_processor.run()
        else:
            self.logger.info("Skip preprocessing data...")
        self.logger.info(f"Start running data module {self.__class__.__name__}...")
        self._run()
        self.logger.info(f"End running data module {self.__class__.__name__}...")


