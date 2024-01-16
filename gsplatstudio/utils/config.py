from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

from omegaconf import OmegaConf
from gsplatstudio.utils.type_utils import *


@dataclass
class ExperimentConfig:
    name: str = "default"
    description: str = ""
    tag: str = ""
    seed: int = 0
    use_timestamp: bool = True
    timestamp: Optional[str] = None
    exp_root_dir: str = "outputs"

    ### these shouldn't be set manually
    exp_dir: str = "outputs/default"
    trial_name: str = "exp"
    trial_dir: str = "outputs/default/exp"
    n_gpus: int = 1
    ###

    data_type: str = ""
    data: dict = field(default_factory=dict)
    system_type: str = ""
    system: dict = field(default_factory=dict)
    checkpoint: dict = field(default_factory=dict)
    logger: dict = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.tag and not self.use_timestamp:
            raise ValueError("Either tag is specified or use_timestamp is True.")
        self.trial_name = self.tag
        # if resume from an existing config, self.timestamp should not be None
        if self.timestamp is None:
            self.timestamp = ""
            if self.use_timestamp:
                self.timestamp = datetime.now().strftime("@%Y%m%d-%H%M%S")
        self.trial_name += self.timestamp
        self.exp_dir = str(Path(self.exp_root_dir) / self.name)
        self.trial_dir = str(Path(self.exp_dir) / self.trial_name)
        
def load_config(*yamls: str, cli_args: list = [], from_string=False, **kwargs) -> Any:
    if from_string:
        yaml_confs = [OmegaConf.create(s) for s in yamls]
    else:
        yaml_confs = [OmegaConf.load(f) for f in yamls]
    cli_conf = OmegaConf.from_cli(cli_args)
    cfg = OmegaConf.merge(*yaml_confs, cli_conf, kwargs)
    OmegaConf.resolve(cfg)
    assert isinstance(cfg, DictConfig)
    scfg = parse_structured(ExperimentConfig, cfg)
    return scfg

def dump_config(path: str, config) -> None:
    with open(path, "w") as fp:
        OmegaConf.save(config=config, f=fp)

def parse_structured(fields: Any, cfg: Optional[Union[dict, DictConfig]] = None) -> Any:
    scfg = OmegaConf.structured(fields(**cfg))
    return scfg




