import cv2

from structure.cv.find_contours import ContoursData
from structure.pipeline import DetectionPass


class FilterByArea(DetectionPass):
    def __init__(self, min_area_ratio=0.00031, max_area_ratio=0.0074):
        super().__init__()
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio

    def run(self, input_data: ContoursData) -> ContoursData:
        height, width = self.get_original_image().shape[:2]
        image_area = width * height

        min_area = image_area * self.min_area_ratio
        max_area = image_area * self.max_area_ratio
        print(max_area)

        filtered_contours = list(filter(
            lambda c: max_area > cv2.contourArea(c) > min_area,
            input_data.contours
        ))

        return ContoursData(filtered_contours)
