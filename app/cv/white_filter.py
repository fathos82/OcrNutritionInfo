import numpy as np

from project.pipeline import DetectionPass, IOComponent
from project.utils.image_data import ImageData

import cv2


class WhiteFilter(DetectionPass):
    def run(self, input_data: ImageData) -> ImageData:
        img = input_data.image
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_values = np.array([37, 0, 131])
        upper_values = np.array([170, 25, 152])
        mask = cv2.inRange(hsv, lower_values, upper_values)
        cv2.imshow('mask', mask)
        return ImageData.from_image(mask)
