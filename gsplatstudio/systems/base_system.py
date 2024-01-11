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

class BaseSystem(ABC):
    def __init__(self, cfg):
        self.cfg = parse_structured(self.config_class, cfg)
        self.model = gsplatstudio.find(self.cfg.representation_type)(self.cfg.representation)
        self.loss = gsplatstudio.find(self.cfg.loss_type)(self.cfg.loss)
        self.trainer = gsplatstudio.find(self.cfg.trainer_type)(self.cfg.trainer)
        self.paramOptim = gsplatstudio.find(self.cfg.paramOptim_type)(self.cfg.paramOptim)
        self.structOptim = gsplatstudio.find(self.cfg.structOptim_type)(self.cfg.structOptim)
        self.renderer = gsplatstudio.find(self.cfg.renderer_type)(self.cfg.renderer)
        
    @property
    def config_class(self):
        return BaseSystemConfig
    
    @property
    @abstractmethod
    def record_cfgs_raw(self):
        return {}
    
    def get_record_cfgs(self, log_dir):
        record_cfgs = []
        for cfg in self.record_cfgs_raw:
            cfg.update({"filepath": log_dir})
            record_cfgs.append(cfg)
        return record_cfgs
    
    def restore(self, data, logger, log_dir, ckpt_dir, system_path, iteration):
        self.logger = logger
        self.logger.info(f"Start restoring system from {system_path} of iteration {iteration}... ")
        
        # init recorder
        self.recorder = Recorder(self.get_record_cfgs(log_dir))
        self.trainer.set("ckpt_dir", ckpt_dir)
        self.trainer.set("view_dir", data.view_dir)
        self.trainer.initialize_components(logger = logger, recorder = self.recorder, 
                    data = data, model = self.model, 
                    loss = self.loss, renderer = self.renderer,
                    paramOptim = self.paramOptim, structOptim = self.structOptim)

        self.trainer.restore_components(system_path, iteration)
        self.logger.info("End restoring system ... ")
        
    def load(self, data, logger, log_dir, ckpt_dir):
        self.logger = logger
        self.logger.info("Start loading system ... ")
        
        # init recorder
        self.recorder = Recorder(self.get_record_cfgs(log_dir))
        self.trainer.set("ckpt_dir", ckpt_dir)
        self.trainer.set("view_dir", data.view_dir)
        self.trainer.initialize_components(logger = logger, recorder = self.recorder, 
                          data = data, model = self.model, 
                          loss = self.loss, renderer = self.renderer,
                          paramOptim = self.paramOptim, structOptim = self.structOptim)
        self.trainer.setup_components()

        
        # Save cfg_args
        namespace_str = f"Namespace(data_device='{data.data_device}', \
                                eval={data.eval}, images='images', \
                                model_path='{str(data.view_dir)}', resolution={data.resolution}, \
                                sh_degree={self.cfg.representation.max_sh_degree}, source_path='{data.source_path}', \
                                white_background={self.cfg.renderer.background_color == [255,255,255]})"
        with open(data.view_dir / "cfg_args", 'w') as cfg_log_f:
            cfg_log_f.write(namespace_str)

        self.logger.info("End loading system ... ")
    
    def run(self):
        self.logger.info("Start running system ... ")
        self.trainer.train()
        self.logger.info("End running system ... ")
