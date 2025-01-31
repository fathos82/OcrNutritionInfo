import cv2

from project.pipeline import DetectionPass, I, O, IOComponent
from project.utils.image_data import ImageData


class ContoursData(IOComponent):
    def __init__(self, contours):
        self.contours = contours


class FindContours(DetectionPass):
    def run(self, input_data: ImageData) -> O:
        bin = input_data.image # TODO: REVISAR ISSO
        # image = self.get_original_image()
        contours = cv2.findContours(bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
        # for contour in contours:
        #     x, y, w, h = cv2.boundingRect(contour)
        #     cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # cv2.imshow("contours", image)
        return ContoursData(contours)