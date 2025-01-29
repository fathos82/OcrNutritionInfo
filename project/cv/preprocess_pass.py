import numpy as np

from project.pipeline import DetectionPass, IOComponent
from project.utils.image_data import ImageData

import cv2


class PreProcessPass(DetectionPass):
    def run(self, input_data: ImageData) -> ImageData: # TODO: >> VERFICAR OS PARAMETROS
        img = input_data.image
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        median = cv2.medianBlur(gray_image, 3)
        alpha = 2.0
        beta = 50
        ajusted_image = cv2.convertScaleAbs(median, alpha=alpha, beta=beta)
        otsu = cv2.threshold(ajusted_image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # TODO: FINALIZE PRE PROCESSING, TRANSFORMATIONS
        return ImageData.from_image(otsu)
