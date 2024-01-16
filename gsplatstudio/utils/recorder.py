from gsplatstudio.utils.type_utils import *
from gsplatstudio.utils.config import parse_structured
import matplotlib.pyplot as plt
from pathlib import Path
import torchvision.transforms as transforms

@dataclass
class RecorderConfig:
    name: str = ""
    save_folder: str = ""
    value_type: str = ""
    plot_type: str = ""
    save_intervals: int = -1
    save_iters: List[int] = field(default_factory=list)


class Recorder:
    def __init__(self,cfg_list):
        self.cfgs: Dict[str, RecorderConfig] = {}
        self.data: Dict[str, List] = {}
        for cfg_raw in cfg_list:
            cfg = parse_structured(RecorderConfig, cfg_raw)
            self.cfgs.update({cfg.name: cfg})
            self.data.update({cfg.name: []})
            
    def init_components(self, log_dir, max_iter, first_iter = 0):
        for cfg in self.cfgs.values():
            # update save_folder
            save_folder = Path(log_dir) / f"{cfg.name}"
            save_folder.mkdir(parents=True, exist_ok=True)
            cfg.update({"save_folder": str(save_folder)})
            
            # update save_iters
            if cfg.save_intervals != -1:
                cfg.update({"save_iters": list(range(first_iter, max_iter + 1, cfg.save_intervals))})

    def snapshot(self, name, value):
        self.data[name].append(value)

    def update(self, iteration):
        for record_var in self.cfgs.keys():
            cfg = self.cfgs[record_var]
            values = self.data[record_var]
            if self.should_process(iteration, save_iters = cfg.save_iters):
                
                # print(f"record_var: {record_var}, iteration: {iteration}, current_index: {current_index}, start_index: {start_index}")
                if cfg.value_type == "number":
                    current_index = cfg.save_iters.index(iteration)
                    start_index = 0 if current_index == 0 else cfg.save_iters[current_index - 1] + 1
                    data_slice = values[start_index:]
                    # save values
                    with open(str(Path(cfg.save_folder) / "value.txt"), "a") as file:
                        for data in data_slice:
                            file.write(str(data) + "\n")
                    # plot
                    if cfg.plot_type == "line_chart":
                        plot_line_chart(cfg, values, cfg.save_folder)
                elif cfg.value_type == "image_tensor":
                    data_slice = values[-1]
                    image_tensor = transforms.ToPILImage()(data_slice.squeeze(0))
                    image_tensor.save(str(Path(cfg.save_folder) / f"{iteration}.png"))

                elif cfg.value_type == "distribution_array":
                    data_slice = values[-1]
                    plot_distribution_chart(cfg, data_slice, cfg.save_folder, iteration)

                elif cfg.value_type == "image_tensor_once":
                    for idx, (filename, value) in enumerate(values):
                        image_tensor = transforms.ToPILImage()(value.squeeze(0))
                        image_tensor.save(str(Path(cfg.save_folder) / f"{filename}.png"))


    def should_process(self, iter, save_iters = None, name = None):
        if name:
            save_iters = self.cfgs[name].save_iters
        return iter in save_iters
        

def plot_distribution_chart(cfg, values, save_folder, iteration):
    plt.figure()
    plt.hist(values, bins=100)
    plt.yscale('log')
    plt.title(f'Distribution Chart of {cfg.name}')
    plt.xlabel(cfg.name)
    plt.ylabel('Frequency')
    plt.savefig(str(Path(save_folder) / f"distribution_chart_{iteration}.png"))
    plt.close()

def plot_line_chart(cfg, values, save_folder):
    plt.plot(values)
    plt.xlabel('Iteration')
    plt.ylabel(cfg.name)
    plt.title(f'Line Chart of {cfg.name}')
    plt.savefig(str(Path(save_folder) / "line_chart.png"))
    plt.close()

