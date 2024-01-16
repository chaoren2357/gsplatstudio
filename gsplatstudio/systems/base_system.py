from abc import abstractmethod, ABC
import gsplatstudio
from gsplatstudio.utils.type_utils import *
from gsplatstudio.utils.config import parse_structured
from gsplatstudio.utils.recorder import Recorder

@dataclass
class BaseSystemConfig:
    representation_type: str
    representation: str
    trainer_type: str
    trainer: str
    paramOptim_type: str
    paramOptim: str
    structOptim_type: str
    structOptim:str
    loss_type: str
    loss: str
    renderer_type: str
    renderer: str
    recorder: str

class BaseSystem(ABC):

    def __init__(self, cfg, logger):
        self.cfg = parse_structured(self.config_class, cfg)
        self.logger = logger
        self.model = gsplatstudio.find(self.cfg.representation_type)(self.cfg.representation, self.logger)
        self.loss = gsplatstudio.find(self.cfg.loss_type)(self.cfg.loss, self.logger)
        self.trainer = gsplatstudio.find(self.cfg.trainer_type)(self.cfg.trainer, self.logger)
        self.paramOptim = gsplatstudio.find(self.cfg.paramOptim_type)(self.cfg.paramOptim, self.logger)
        self.structOptim = gsplatstudio.find(self.cfg.structOptim_type)(self.cfg.structOptim, self.logger)
        self.renderer = gsplatstudio.find(self.cfg.renderer_type)(self.cfg.renderer, self.logger)
        self.recorder = Recorder(self.cfg.recorder)
    
    @property
    @abstractmethod
    def config_class(self):
        return BaseSystemConfig
    
    def _restore(self, data, system_path, iteration, dirs):
        self.recorder.init_components(log_dir = dirs["log_dir"], max_iter = self.trainer.cfg.iterations)
        self.trainer.init_components(
                    recorder = self.recorder, 
                    data = data, model = self.model, 
                    loss = self.loss, renderer = self.renderer,
                    paramOptim = self.paramOptim, structOptim = self.structOptim,
                    dirs = dirs)
        self.trainer.restore_components(system_path, iteration)

    def _load(self, data, dirs):
        self.recorder.init_components(log_dir = dirs["log_dir"], max_iter = self.trainer.cfg.iterations)
        self.trainer.init_components(
                    recorder = self.recorder, 
                    data = data, model = self.model, 
                    loss = self.loss, renderer = self.renderer,
                    paramOptim = self.paramOptim, structOptim = self.structOptim,
                    dirs = dirs)
        self.trainer.setup_components()
        # Save cfg_args
        namespace_str = f"Namespace(data_device='{data.cfg.device}', \
                                eval={data.cfg.eval}, images='images', \
                                model_path='{str(data.view_dir)}', resolution={data.cfg.resolution}, \
                                sh_degree={self.cfg.representation.max_sh_degree}, source_path='{data.cfg.source_path}', \
                                white_background={self.cfg.renderer.background_color == [255,255,255]})"
        with open(data.view_dir / "cfg_args", 'w') as cfg_log_f:
            cfg_log_f.write(namespace_str)

    def _run(self):
        self.trainer.train()
        
    def restore(self, data, system_path, iteration, dirs):
        assert "ckpt_dir" in dirs and "view_dir" in dirs and "log_dir" in dirs
        self.logger.info(f"Start restoring {self.__class__.__name__} from {system_path} of iteration {iteration}... ")
        self._restore(data, system_path, iteration, dirs)
        self.logger.info(f"End restoring {self.__class__.__name__} ... ")
    
    def load(self, data, dirs):
        assert "ckpt_dir" in dirs and "view_dir" in dirs and "log_dir" in dirs
        self.logger.info(f"Start loading {self.__class__.__name__} ... ")
        self._load(data, dirs)
        self.logger.info(f"End loading {self.__class__.__name__} ... ")
    
    def run(self):
        self.logger.info(f"Start running {self.__class__.__name__} ... ")
        self._run()
        self.logger.info(f"End running {self.__class__.__name__} ... ")
