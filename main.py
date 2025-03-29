from app.app_runner import AppRunner
from project.application import Application
from project.pipeline_factory import PipelineVariation
from project.runner_configuration import RunnerConfiguration
from server.server_application_runner import ServerApplicationRunner

config = RunnerConfiguration(pipeline_variation=PipelineVariation.PIPELINE_1, processing_name='Teste 01')
Application(ServerApplicationRunner(config)).run()
