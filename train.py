from gsplatstudio.systems.experiment import Experiment
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the training experiment.")
    parser.add_argument('-c', '--config', default='configs/gsplat_vanilla.yaml', help='Path to the configuration file.')
    e = Experiment(config_path = parser.parse_args().config)
    e.run()
    
