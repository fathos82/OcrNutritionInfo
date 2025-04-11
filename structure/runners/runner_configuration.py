

class RunnerConfiguration:
    def __init__(self, **kwargs):
        self.processing_name:str = kwargs.get('processing_name')
        self.pipeline = kwargs.get('pipeline')
        self.runner = kwargs.get('runner')


