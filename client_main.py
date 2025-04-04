from app.app_runner import AppRunner
from structure.application import Application
from structure.pipeline_factory import PipelineVariation
from structure.runner_configuration import RunnerConfiguration
from server.client_runner import ClientRunner
config = RunnerConfiguration(pipeline_variation=PipelineVariation.PIPELINE_OCR_5, processing_name='Teste 01')
Application(ClientRunner(config)).run()
