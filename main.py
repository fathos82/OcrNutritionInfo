from app.app_runner import AppRunner
from structure.application import Application
from structure.pipeline_factory import PipelineVariation
from structure.runner_configuration import RunnerConfiguration
from server.server_application_runner import ServerApplicationRunner

config = RunnerConfiguration(pipeline_variation=PipelineVariation.PIPELINE_1, processing_name='Teste 01')
Application(ServerApplicationRunner(config)).run()
