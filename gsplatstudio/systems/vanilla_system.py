import gsplatstudio
from gsplatstudio.systems.base_system import BaseSystem
from gsplatstudio.utils.type_utils import *

@dataclass
class VanillaSystemConfig:
    representations: list = field(default_factory=list)
    trainers: list = field(default_factory=list)
    paramOptims: list = field(default_factory=list)
    structOptims:list = field(default_factory=list)
    losses: list = field(default_factory=list)
    renderers: list = field(default_factory=list)
    recorder: list = field(default_factory=list)


@gsplatstudio.register("vanilla-system")
class VanillaSystem(BaseSystem):
    @property
    def config_class(self):
        return VanillaSystemConfig
    
    def _load(self, data, dirs):
        self.data = data
        self.recorder.init_components(log_dir = dirs["log_dir"], 
                                      max_iter = self.trainer.cfg.iterations + self.trainer2.cfg.iterations)
        self.trainer.init_components(recorder = self.recorder, progress_bar = self.progress_bar, dirs = dirs)
        self.trainer.setup_components(data=self.data, representation=self.representation, 
                                      loss=self.loss, paramOptim=self.paramOptim, 
                                      renderer=self.renderer, structOptim=self.structOptim, 
                                      first_iteration = 1)
        self.trainer2.init_components(recorder = self.recorder, progress_bar = self.progress_bar, dirs = dirs)
        self.trainer2.setup_components(data=self.data, representation=self.representation, 
                                      loss=self.loss, paramOptim=self.paramOptim, 
                                      renderer=self.renderer, structOptim=self.structOptim, 
                                      first_iteration=self.trainer.cfg.iterations + 1)
        # Save cfg_args
        namespace_str = f"Namespace(data_device='{data.cfg.device}', \
                                eval={data.cfg.eval}, images='images', \
                                model_path='{str(data.view_dir)}', resolution={data.cfg.resolution}, \
                                sh_degree={self.cfg.representations[0].params.max_sh_degree}, source_path='{data.cfg.source_path}', \
                                white_background={self.cfg.renderers[0].params.background_color == [255,255,255]})"
        with open(data.view_dir / "cfg_args", 'w') as cfg_log_f:
            cfg_log_f.write(namespace_str)
    
    def _run(self):
        self.trainer.train()
        self.trainer2.train()