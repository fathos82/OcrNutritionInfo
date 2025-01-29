import numpy as np

from project.pipeline import IOComponent

import cv2
class ImageData(IOComponent):
    #TODO: Temporary Constructor
    def __init__(self, path:str):
        if path is not None:
            self.image = cv2.imread(path)

    @classmethod
    def from_image(cls, image:np.ndarray):
        self = cls(None)
        self.image = image
        return self
    def __str__(self):
        return f"ImageData(image={self.image.shape})"