import torch
import shutil
import random
import numpy as np
from pathlib import Path
import gsplatstudio
from gsplatstudio.utils.config import load_config, dump_config
from gsplatstudio.utils.system_utils import copy_items
from gsplatstudio.utils.logger import setup_logging, init_logger

class Experiment:
    def __init__(self, config_path='configs/gsplat_vanilla.yaml'):
        self.cfg = load_config(config_path)
        self._init_seed(self.cfg.seed)

        self._prepare_experiment_folders(config_path)
        self.logger.info("Experiment folder: {}".format(self.cfg.trial_dir))

        self.data = gsplatstudio.find(self.cfg.data_type)(self.cfg.data, self.logger, {"view_dir": self.view_dir})
        self.system = gsplatstudio.find(self.cfg.system_type)(self.cfg.system, self.logger)
        self.cfg_ckpt =  self.cfg.checkpoint
        self.logger.info("Finish experiment initialization...")

    def _prepare_experiment_folders(self,config_path):
        # Code duplicate
        excluded_dirs = {'configs', 'outputs', 'viewer', '__pycache__', 'submodules', 'docs', 'assets'}
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
        self.logger.info("Start running experiment...")
        if self.cfg_ckpt.use:
            self.data.restore(self.cfg_ckpt.data_path, self.cfg_ckpt.iteration)
            self.system.restore(self.data, self.cfg_ckpt.system_path, self.cfg_ckpt.iteration, 
                                dirs = {
                                            "log_dir":self.log_dir, 
                                            "ckpt_dir": self.ckpt_dir, 
                                            "view_dir": self.data.view_dir
                                        })
            self.system.run()
        else:
            self.data.run()
            self.system.load(self.data,
                             dirs = {
                                        "log_dir":self.log_dir, 
                                        "ckpt_dir": self.ckpt_dir, 
                                        "view_dir": self.data.view_dir
                                    })
            self.system.run()
        self.logger.info("Finish running experiment...")

