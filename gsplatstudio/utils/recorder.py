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
        
    def init_components(self, log_dir, max_iter, first_iter=0):
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
                if cfg.value_type == "number":
                    self.store_number(name = record_var, data = values, iteration = iteration, save_folder = cfg.save_folder)
                elif cfg.value_type == "1darray":
                    self.store_1darray(name = record_var, data = values, iteration = iteration, save_folder = cfg.save_folder)
                elif cfg.value_type == "imageTensorDict":
                    self.store_imageTensorDict(name = record_var, data = values, iteration = iteration, save_folder = cfg.save_folder)
                self.data[record_var] = [] # clear temporary data

    def should_process(self, iter, save_iters = None, name = None):
        if name:
            save_iters = self.cfgs[name].save_iters
        return iter in save_iters
    
    @staticmethod
    def store_number(name, data, iteration, save_folder):
        # save values
        save_file = Path(save_folder) / "value.txt"
        with open(str(save_file), "a") as file:
            for d in data:
                file.write(str(d) + "\n")

        # plot line chart
        save_chart = Path(save_folder) / "line_chart.png"
        values = []
        with open(save_file, 'r') as file:
            for line in file:
                try:
                    value = float(line.strip())
                    values.append(value)
                except ValueError:
                    print(f"Skipping invalid line: {line}")

        plt.plot(values)
        plt.xlabel('Iteration')
        plt.ylabel(name)
        plt.title(f'Line Chart of {name}')
        plt.savefig(str(save_chart))
        plt.close()
    
    @staticmethod
    def store_1darray(name, data, iteration, save_folder):
        plt.figure()
        plt.hist(data, bins=100)
        plt.yscale('log')
        plt.title(f'Distribution Chart of {name}')
        plt.xlabel(name)
        plt.ylabel('Frequency')
        plt.savefig(str(Path(save_folder) / f"distribution_chart_{iteration}.png"))
        plt.close()

    @staticmethod
    def store_imageTensorDict(name, data, iteration, save_folder):
        for idx, (filename, value) in enumerate(data):
            image_tensor = transforms.ToPILImage()(value.squeeze(0))
            image_tensor.save(str(Path(save_folder) / f"{filename}.png"))

