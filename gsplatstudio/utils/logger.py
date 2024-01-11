import logging
import logging.config
from pathlib import Path
from omegaconf import OmegaConf
VERBOSE = 15
def verbose(self, message, *args, **kws):
    if self.isEnabledFor(VERBOSE):
        # Yes, logger takes its '*args' as 'args'.
        self._log(VERBOSE, message, args, **kws) 
logging.addLevelName(VERBOSE, 'VERBOSE')
logging.Logger.verbose = verbose

def setup_logging(logger_cfg, logger_path):
    """
    Setup logging configuration
    """
    logger_cfg.handlers.info_file_handler.filename = str(logger_path / logger_cfg.handlers.info_file_handler.filename)
    logging.config.dictConfig(OmegaConf.to_container(logger_cfg, resolve=True))

def init_logger(name):
    logger = logging.getLogger(name)
    return logger