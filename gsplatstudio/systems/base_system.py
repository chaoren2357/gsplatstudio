from abc import abstractmethod, ABC
import gsplatstudio
from gsplatstudio.utils.type_utils import *
from gsplatstudio.utils.config import parse_structured
from gsplatstudio.utils.recorder import Recorder

@dataclass
class BaseSystemConfig:
    representations: list = field(default_factory=list)
    trainers: list = field(default_factory=list)
    paramOptims: list = field(default_factory=list)
    structOptims:list = field(default_factory=list)
    losses: list = field(default_factory=list)
    renderers: list = field(default_factory=list)
    recorder: str

class BaseSystem(ABC):

    def __init__(self, cfg, logger):
        self.cfg = parse_structured(self.config_class, cfg)
        self.logger = logger
        for representation_cfg in self.cfg.representations:
            representation = gsplatstudio.find(representation_cfg.type)(representation_cfg.params, self.logger)
            setattr(self, representation_cfg.id, representation)
        for loss_cfg in self.cfg.losses:
            loss = gsplatstudio.find(loss_cfg.type)(loss_cfg.params, self.logger)
            setattr(self, loss_cfg.id, loss)
        for trainer_cfg in self.cfg.trainers:
            trainer = gsplatstudio.find(trainer_cfg.type)(trainer_cfg.params, self.logger)
            setattr(self, trainer_cfg.id, trainer)
        for paramOptim_cfg in self.cfg.paramOptims:
            paramOptim = gsplatstudio.find(paramOptim_cfg.type)(paramOptim_cfg.params, self.logger)
            setattr(self, paramOptim_cfg.id, paramOptim)
        for structOptim_cfg in self.cfg.structOptims:
            structOptim = gsplatstudio.find(structOptim_cfg.type)(structOptim_cfg.params, self.logger)
            setattr(self, structOptim_cfg.id, structOptim)
        for renderer_cfg in self.cfg.renderers:
            renderer = gsplatstudio.find(renderer_cfg.type)(renderer_cfg.params, self.logger)
            setattr(self, renderer_cfg.id, renderer)

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
        self.data = data
        self.recorder.init_components(log_dir = dirs["log_dir"], max_iter = self.trainer.cfg.iterations)
        self.trainer.init_components(recorder = self.recorder, dirs = dirs)
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
