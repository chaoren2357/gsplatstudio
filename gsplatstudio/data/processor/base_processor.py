import subprocess
from abc import abstractmethod, ABC

import gsplatstudio
from gsplatstudio.utils.type_utils import *
from gsplatstudio.utils.config import parse_structured

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
    @abstractmethod
    def should_skip(self):
        pass
    
    @abstractmethod
    def _run(self):
        pass

    def run(self):
        self.logger.info(f"Start running data-processor {self.__class__.__name__}...")
        self._run()
        self.logger.info(f"End running data-processor {self.__class__.__name__}...")

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
    pass

@gsplatstudio.register("no-processor")
class NoDataProcessor(BaseDataProcessor):
    def __init__(self, cfg, logger, source_path) -> None:
        self.logger = logger
    @property
    def config_class(self):
        return NoDataProcessorConfig

    @property
    def should_skip(self):
        return True
    
    def _run(self):
        pass
