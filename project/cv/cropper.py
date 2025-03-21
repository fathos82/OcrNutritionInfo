import cv2
import numpy as np

from project.pipeline import DetectionPass
from project.utils.image_data import ImageData


class Cropper(DetectionPass):
    def __init__(self, crop_factor:float|tuple[float, float]=1):
        super(Cropper, self).__init__()
        if isinstance(crop_factor, tuple):
            self.cf_x = crop_factor[0]
            self.cf_y = crop_factor[1]
        else:
            self.cf_x = crop_factor
            self.cf_y = crop_factor


        if self.cf_x < 0 or self.cf_x > 1 and self.cf_y < 0 or self.cf_y > 1 :
            raise ValueError("'crop_factor' must be a number between 0 and 1'  ")
    def run(self, input_data: ImageData) -> ImageData:
        img = input_data.image
        new_y =  int((1- self.cf_y) * img.shape[0] // 2)
        new_x = int((1 - self.cf_x) * img.shape[1] // 2)

        new_image = img[new_y : img.shape[0] - new_y, new_x :img.shape[1] - new_x]
        # cv2.imshow("Cropped", new_image)
        self.original_image.from_image(new_image)
        return ImageData.from_image(new_image)
