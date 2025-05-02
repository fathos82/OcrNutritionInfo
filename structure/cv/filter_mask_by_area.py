import numpy as np

from structure.pipeline import DetectionPass, IOComponent
from structure.utils.image_data import ImageData

import cv2


class FilterMaskByArea(DetectionPass):

    def __init__(self, minArea=150):
        super().__init__()
        self.minArea = minArea
    def run(self, input_data: ImageData) -> ImageData:
        mask = input_data.image
        componentsNumber, labeledImage, componentStats, componentCentroids = cv2.connectedComponentsWithStats(mask, connectivity=4)

        # Get the indices/labels of the remaining components based on the area stat
        # (skip the background component at index 0)
        remainingComponentLabels = [i for i in range(1, componentsNumber) if componentStats[i][4] >= self.minArea]

        # Filter the labeled pixels based on the remaining labels,
        # assign pixel intensity to 255 (uint8) for the remaining pixels
        filteredImage = np.where(np.isin(labeledImage, remainingComponentLabels) == True, 255, 0).astype('uint8')
        # cv2.imshow('filteredImage', filteredImage)
        return ImageData.from_image(mask)
