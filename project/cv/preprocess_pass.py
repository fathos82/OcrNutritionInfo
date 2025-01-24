from project.pipeline import DetectionPass, IOComponent
from project.utils.image_data import ImageData

from project.pipeline import IOComponent


class ProcessedImage(IOComponent):
    def __init__(self, image):
        self.image = image



class PreProcessPass(DetectionPass):
    def run(self, input_data:ImageData)->ProcessedImage:
        img = input_data.image
        print("Processing Image")
        print("Image Size: "+str(img.shape))

        return ProcessedImage(None)

    def __init__(self):
        pass