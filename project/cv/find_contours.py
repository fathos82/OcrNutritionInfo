import cv2

from project.pipeline import DetectionPass, I, O, IOComponent
from project.utils.image_data import ImageData


class ContoursData(IOComponent):
    def __init__(self, contours, temporary_image):
        self.temporary_image = temporary_image
        self.contours = contours


class FindContours(DetectionPass):
    def run(self, input_data: ImageData) -> O:
        bin = input_data.image # TODO: REVISAR ISSO
        contours = cv2.findContours(bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
        img = self.get_original_image()

        return ContoursData(contours, img)