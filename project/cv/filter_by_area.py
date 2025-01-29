import cv2

from project.cv.find_contours import ContoursData
from project.pipeline import DetectionPass, O


class FilterByArea(DetectionPass):
    def __init__(self, min_area=250, max_area=900):
        super().__init__()
        self.min_area = min_area
        self.max_area = max_area

    def run(self, input_data: ContoursData) -> ContoursData:
        filtered_contours = list(filter(lambda c : self.max_area > cv2.contourArea(c) > self.min_area,input_data.contours))
        img = self.get_original_image()
        return ContoursData(filtered_contours, img)