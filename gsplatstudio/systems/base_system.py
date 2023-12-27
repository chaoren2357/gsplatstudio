import gsplatstudio
from gsplatstudio.utils.type_utils import *
from gsplatstudio.utils.config import parse_structured
from abc import ABC


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

class BaseSystem:
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
    
    def load(self, data, logger):
        self.trainer.load(logger = logger, data = data, model = self.model, 
                          loss = self.loss, renderer = self.renderer,
                          paramOptim = self.paramOptim, structOptim = self.structOptim)
    def run(self):
        self.trainer.train()
