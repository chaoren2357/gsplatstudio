from gsplatstudio.utils.type_utils import *
from gsplatstudio.utils.config import parse_structured
import matplotlib.pyplot as plt
from pathlib import Path

@dataclass
class RecorderConfig:
    name: str = ""
    filepath: str = ""
    plot_type: str = "line_chart"
    plot_iters: int = 100
    data_values: list = field(default_factory=list)


class Recorder:
    def __init__(self,cfg_list):
        self.cfgs = {}
        for cfg in cfg_list:
            cfg_parsed = parse_structured(RecorderConfig, cfg)
            self.cfgs.update({cfg_parsed.name: cfg_parsed})
            txt_path = Path(cfg_parsed.filepath) / f"{cfg_parsed.name}.txt"
            txt_path.parent.mkdir(parents=True, exist_ok=True)
        self.iter = 0

    def snapshot(self,name,value):
        self.cfgs[name].data_values.append(value)
    def update(self):
        self.iter += 1
        self._save()
    def _save(self):
        for cfg in self.cfgs.values():
            if self.iter % cfg.plot_iters == 0:
                # save values
                with open(f"{cfg.filepath}/{cfg.name}.txt", "a") as file:
                    data_to_write = cfg.data_values[-cfg.plot_iters:]
                    for data in data_to_write:
                        file.write(str(data) + "\n")
                # plot
                if cfg.plot_type == "line_chart":
                    plot_line_chart(cfg)



def plot_line_chart(cfg):
    plt.plot(cfg.data_values)
    plt.xlabel('Iteration')
    plt.ylabel(cfg.name)
    plt.title(f'Line Chart of {cfg.name}')
    plt.savefig(f"{cfg.filepath}/{cfg.name}.png")
    plt.close()

