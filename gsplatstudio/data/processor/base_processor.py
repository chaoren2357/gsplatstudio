from abc import abstractmethod, ABC
from gsplatstudio.utils.config import parse_structured
import gsplatstudio
from gsplatstudio.utils.type_utils import *

class BaseDataProcessor(ABC):
    def __init__(self, cfg, logger, source_path) -> None:
        self.cfg = parse_structured(self.config_class, cfg)
        self.logger = logger
        self.source_path_str = source_path
        
    @property
    @abstractmethod
    def config_class(self):
        pass

    @abstractmethod
    def run(self):
        pass


@dataclass
class NoDataProcessorConfig:
    use_gpu: bool = True
    skip_matching: bool = False
    camera: str = "OPENCV"
    map_ba_global_function_tolerance: float = 0.000001

@gsplatstudio.register("no-processor")
class NoDataProcessor(BaseDataProcessor):
    def __init__(self, cfg, logger, source_path) -> None:
        super().__init__(cfg, logger, source_path)
    @property
    def config_class(self):
        return NoDataProcessorConfig
    def run(self):
        pass
