import torch
import shutil
import random
import numpy as np
from pathlib import Path
import gsplatstudio
from gsplatstudio.utils.config import load_config, dump_config
from gsplatstudio.utils.general_utils import copy_items
from gsplatstudio.utils.logger import setup_logging, init_logger

class Experiment:
    def __init__(self, config_path='configs/gsplat_vanilla.yaml'):
        self.cfg = load_config(config_path)
        self._init_seed(self.cfg.seed)

        self._prepare_experiment_folders(config_path)
        self.logger.info("Experiment folder: {}".format(self.cfg.trial_dir))

        self.data = gsplatstudio.find(self.cfg.data_type)(self.cfg.data, self.logger, Path(self.cfg.trial_dir) / 'results')
        self.system = gsplatstudio.find(self.cfg.system_type)(self.cfg.system)  
        self.logger.info("Finish experiment initialization...")

    def _prepare_experiment_folders(self,config_path):
        # Code duplicate
        excluded_dirs = {'configs', 'outputs', 'viewer', '__pycache__', 'submodules'}
        excluded_files = {'README.md'}
        self.code_dir = Path(self.cfg.trial_dir) / 'code'
        self.code_dir.mkdir(parents=True, exist_ok=True)
        current_dir = Path('.')
        copy_items(current_dir, self.code_dir, excluded_dirs, excluded_files)
        
        # Config duplicate
        self.config_dir = Path(self.cfg.trial_dir) / 'config'
        self.config_dir.mkdir(parents=True, exist_ok=True)
        dump_config(self.config_dir / "parsed.yaml", self.cfg)
        shutil.copyfile(config_path, self.config_dir / "raw.yaml")

        # logs
        self.log_dir = Path(self.cfg.trial_dir) / 'logs'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(self.cfg.logger, self.log_dir)
        self.logger =  init_logger(self.cfg.trial_name)
        # render results
        self.view_dir = Path(self.cfg.trial_dir) / 'results'
        self.view_dir.mkdir(parents=True, exist_ok=True)
        namespace_str = f"Namespace(data_device='{self.cfg.data.data_device}', \
                          eval={self.cfg.data.eval}, images='images', \
                          model_path='{str(self.view_dir)}', resolution={self.cfg.data.resolution}, \
                          sh_degree={self.cfg.system.representation.max_sh_degree}, source_path='{self.cfg.data.source_path}', \
                          white_background={self.cfg.system.renderer.background_color == [255,255,255]})"
        with open(self.view_dir / "cfg_args", 'w') as cfg_log_f:
            cfg_log_f.write(namespace_str)

        # ckpts
        self.ckpt_dir = Path(self.cfg.trial_dir) / 'ckpts'
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _init_seed(self,seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def run(self):
        self.logger.info("Start data initialization...")
        self.data.run()
        self.logger.info("End data initialization...")

        self.logger.info("Start system loading...")
        self.system.load(self.data, self.logger, self.log_dir)
        self.logger.info("End system loading...")

        self.logger.info("Start system running...")
        self.system.run()
        self.logger.info("End system running...")
