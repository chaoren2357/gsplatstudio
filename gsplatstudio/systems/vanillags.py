import gsplatstudio
from gsplatstudio.systems.base_system import BaseSystem

@gsplatstudio.register("vanilla-gsplat")
class vanillaGS(BaseSystem):
    def __init__(self, cfg):
        super().__init__(cfg)
    
    @property
    def record_cfgs_raw(self):
        return [
                {"name": "ema_loss_for_log", "plot_type": "line_chart", "plot_iters": 1000},
                {"name": "loss", "plot_type": "line_chart", "plot_iters": 1000}
            ]        