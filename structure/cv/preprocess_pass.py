import numpy as np

from structure.pipeline import DetectionPass, IOComponent
from structure.utils.image_data import ImageData

import cv2


class PreProcessPass(DetectionPass):
    def run(self, input_data: ImageData) -> ImageData: # TODO: >> VERFICAR OS PARAMETROS
        img = input_data.image
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        median = cv2.medianBlur(gray_image, 3)
        alpha = 2.0
        beta = 50
        ajusted_image = cv2.convertScaleAbs(median, alpha=alpha, beta=beta)
        otsu = cv2.threshold(ajusted_image, 127, 255, cv2.THRESH_TOZERO)[1]
        inv = cv2.bitwise_not(otsu)
        kernel = np.ones((2, 2), np.uint8)
        transformed = cv2.dilate(inv, kernel, iterations=3)
        otsu = cv2.bitwise_not(transformed)
        # cv2.imshow('otsu', otsu)
        # TODO: FINALIZE PRE PROCESSING, TRANSFORMATIONS
        return ImageData.from_image(otsu)
