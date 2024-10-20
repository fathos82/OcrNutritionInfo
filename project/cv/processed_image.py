from project.pipeline import IOComponent


class ProcessedImage(IOComponent):
    def __init__(self, image):
        self.image = image
