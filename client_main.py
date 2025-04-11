from main import RunnerConfiguration
from structure.application import Application
from structure.pipeline_factory import PipelineVariation
from structure.runners.runner_factory import RunnerVariation
from structure.runners.server.client_runner import ClientRunner

config = RunnerConfiguration(pipeline=PipelineVariation.PIPELINE_OCR_6, runner=RunnerVariation.CLIENT)
Application(config).run()
