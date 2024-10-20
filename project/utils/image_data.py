from project.pipeline import IOComponent

import cv2
class ImageData(IOComponent):
    #TODO: Temporary Constructor
    def __init__(self, path:str):
        self.image = cv2.imread(path)