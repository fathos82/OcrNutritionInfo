from enum import Enum

from structure.runners.local.app_runner import LocalRunner
from structure.runners.server.client_runner import ClientRunner
from structure.runners.server.server_application_runner import ServerApplicationRunner


class RunnerVariation(Enum):
    LOCAL = 'local'
    SERVER = 'server'
    CLIENT = 'client'

class RunnerFactory:
    @staticmethod
    def create(configs):
        runner = RunnerVariation(configs.runner)
        match runner:
            case RunnerVariation.LOCAL:
                return LocalRunner(configs)
            case RunnerVariation.SERVER:
                return ServerApplicationRunner(configs)
            case RunnerVariation.CLIENT:
                return ClientRunner(configs)
            case _:
                raise NotImplementedError()
