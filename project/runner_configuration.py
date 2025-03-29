class RunnerConfiguration:
    def __init__(self, **kwargs):
        self.processing_name = kwargs.get('processing_name')
        self.pipeline_variation = kwargs.get('pipeline_variation')