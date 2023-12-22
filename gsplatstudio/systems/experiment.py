import gsplatstudio
import shutil
import numpy as np
import random
import torch
from pathlib import Path
from gsplatstudio.utils.config import load_config, dump_config
from gsplatstudio.utils.general_utils import copy_items
from gsplatstudio.utils.logger import setup_logging, init_logger

class Experiment:
    def __init__(self, config_path='configs/gsplat_vanilla.yaml'):
        self.cfg = load_config(config_path)

        self._init_seed(self.cfg.seed)
        self._prepare_experiment_folders(config_path)
        self.logger.info("Experiment folder: {}".format(self.cfg.trial_dir))
        
        self.logger.info("Loading components...")
        self.data = gsplatstudio.find(self.cfg.data_type)(self.cfg.data)
        self.system = gsplatstudio.find(self.cfg.system_type)()  
        self.model = gsplatstudio.find(self.cfg.system.representation_type)(self.cfg.system.representation)
        self.loss = gsplatstudio.find(self.cfg.system.loss_type)(self.cfg.system.loss)
        self.trainer = gsplatstudio.find(self.cfg.system.trainer_type)(self.cfg.system.trainer)
        self.paramOptim = gsplatstudio.find(self.cfg.system.paramOptim_type)(self.cfg.system.paramOptim)
        self.structOptim = gsplatstudio.find(self.cfg.system.structOptim_type)(self.cfg.system.structOptim)
        self.renderer = gsplatstudio.find(self.cfg.system.renderer_type)(self.cfg.system.renderer)
        self.trainer.load(logger = self.logger, data = self.data, model = self.model, 
                          loss = self.loss, renderer = self.renderer,
                          paramOptim = self.paramOptim, structOptim = self.structOptim)
        

    def _prepare_experiment_folders(self,config_path):
        # Code duplicate
        excluded_dirs = {'configs', 'outputs', 'viewer', '__pycache__', 'submodules'}
        excluded_files = {'README.md'}
        code_dir = Path(self.cfg.trial_dir) / 'code'
        code_dir.mkdir(parents=True, exist_ok=True)
        current_dir = Path('.')
        copy_items(current_dir, code_dir, excluded_dirs, excluded_files)
        
        # Config duplicate
        config_dir = Path(self.cfg.trial_dir) / 'config'
        config_dir.mkdir(parents=True, exist_ok=True)
        dump_config(config_dir / "parsed.yaml", self.cfg)
        shutil.copyfile(config_path, config_dir / "raw.yaml")

        # logs
        log_dir = Path(self.cfg.trial_dir) / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(self.cfg.logger, log_dir)
        self.logger =  init_logger(self.cfg.trial_name)

        
        # TODO:ckpts
        # self.cfg.ckpt_dir = Path(self.cfg.trial_dir) / 'ckpts'


        # TODO:render results
        # self.cfg.results_dir = Path(self.cfg.trial_dir) / 'results'

        # Add cfg file for the viewer(will remove in the future)
        self.cfg.data.model_path = self.cfg.trial_dir
        namespace_str = f"Namespace(data_device='{self.cfg.data.data_device}', \
                          eval={self.cfg.data.eval}, images='{self.cfg.data.images}', \
                          model_path='{self.cfg.data.model_path}', resolution={self.cfg.data.resolution}, \
                          sh_degree={self.cfg.system.representation.max_sh_degree}, source_path='{self.cfg.data.source_path}', \
                          white_background={self.cfg.system.renderer.background_color == [255,255,255]})"
        with open(Path(self.cfg.data.model_path) / "cfg_args", 'w') as cfg_log_f:
            cfg_log_f.write(namespace_str)


    def _init_seed(self,seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    def run(self):
        self.trainer.train()
        # self.system.run()