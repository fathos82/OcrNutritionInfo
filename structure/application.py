from structure.runners.runner_factory import RunnerFactory


class Application:
    def __init__(self, configs ):
        self.runner = RunnerFactory.create(configs)
        pass
    def run(self):
        self.runner.run()