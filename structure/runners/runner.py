from abc import abstractmethod, ABC

from structure.pipeline_factory import PipelineFactory, PipelineVariation
from structure.runners.runner_configuration import RunnerConfiguration


class Runner(ABC):
    def __init__(self, config:RunnerConfiguration):
        self.pipeline = PipelineFactory.create_pipeline(PipelineVariation(config.pipeline))
        self.processing_name = config.processing_name
    @abstractmethod
    def run(self):
        pass


