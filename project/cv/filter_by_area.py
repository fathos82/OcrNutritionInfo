import cv2

from project.cv.find_contours import ContoursData
from project.pipeline import DetectionPass, I, O


class FilterByArea(DetectionPass):
    def __init__(self, min_area=100, max_area=2000):
        super().__init__()
        self.min_area = min_area
        self.max_area = max_area

    def run(self, input_data: ContoursData) -> O:
        filtered_contours = filter(lambda c : self.max_area > cv2.contourArea(c) > self.min_area,input_data.contours)
        img = self.get_original_image()
        for contour in filtered_contours:
            x, y, w, h = cv2.boundingRect(contour)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return ContoursData(filtered_contours, img)