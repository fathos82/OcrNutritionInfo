from project.cv.processed_image import ProcessedImage
from project.pipeline import DetectionPass, IOComponent
from project.utils.image_data import ImageData




class PreProcessPass(DetectionPass):
    def run(self, input_data:ImageData)->ProcessedImage:
        img = input_data.image
        print("Processing Image")
        print("Image Size: "+str(img.shape))

        return ProcessedImage(None)

    def __init__(self):
        pass