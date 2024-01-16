import gsplatstudio
from gsplatstudio.systems.base_system import BaseSystem
from gsplatstudio.utils.type_utils import *

@dataclass
class VanillaSystemConfig:
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


@gsplatstudio.register("vanilla-system")
class VanillaSystem(BaseSystem):
    @property
    def config_class(self):
        return VanillaSystemConfig
    