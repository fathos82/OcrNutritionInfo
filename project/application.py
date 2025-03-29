from project.runner import Runner


class Application:
    def __init__(self, runner: Runner):
        self.runner = runner
    def run(self):
        self.runner.run()