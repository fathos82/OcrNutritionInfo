from abc import abstractmethod, ABC

from project.runner_configuration import RunnerConfiguration


class Runner(ABC):
    def __init__(self, config: RunnerConfiguration):
        self.config = config
    @abstractmethod
    def run(self):
        pass


