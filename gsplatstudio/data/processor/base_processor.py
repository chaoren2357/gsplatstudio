from abc import abstractmethod, ABC
import subprocess
from gsplatstudio.utils.config import parse_structured
import gsplatstudio
from gsplatstudio.utils.type_utils import *

class BaseDataProcessor(ABC):
    def __init__(self, cfg, logger, source_path) -> None:
        self.cfg = parse_structured(self.config_class, cfg)
        self.logger = logger
        self.source_path_str = source_path
        
    @property
    @abstractmethod
    def config_class(self):
        pass

    @property
    def should_skip(self):
        pass

    @abstractmethod
    def run(self):
        pass

    def run_command_with_realtime_output(self, cmd):
        """
        Run the specified command and output the results in real-time.

        :param cmd: The command string to run.
        :return: The exit code of the command.
        """
        self.logger.info(f"Running command: {cmd}")
        # Start the process
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Read output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                self.logger.verbose(output.strip())  

        # Read any remaining error output
        stderr_output = process.stderr.read()
        if stderr_output:
            self.logger.error("Error Output:")  
            self.logger.error(stderr_output.strip())

        # Return the exit code
        return process.returncode

@dataclass
class NoDataProcessorConfig:
    use_gpu: bool = True
    camera: str = "OPENCV"
    map_ba_global_function_tolerance: float = 0.000001

@gsplatstudio.register("no-processor")
class NoDataProcessor(BaseDataProcessor):
    def __init__(self, cfg, logger, source_path) -> None:
        super().__init__(cfg, logger, source_path)
    @property
    def config_class(self):
        return NoDataProcessorConfig

    @property
    def should_skip(self):
        return True
    
    def run(self):
        pass
