import gsplatstudio
from gsplatstudio.systems.base_system import BaseSystem

@gsplatstudio.register("vanilla-gsplat")
class vanillaGS(BaseSystem):
    def __init__(self, cfg):
        super().__init__(cfg)