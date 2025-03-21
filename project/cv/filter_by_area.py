import cv2

from project.cv.find_contours import ContoursData
from project.pipeline import DetectionPass, O, IOComponent
from project.utils.image_data import ImageData




class FilterByArea(DetectionPass):
    def __init__(self, min_area=200, max_area=999999999):
        super().__init__()
        self.min_area = min_area
        self.max_area = max_area

    def run(self, input_data: ContoursData) -> ContoursData:
        filtered_contours = list(filter(lambda c : self.max_area > cv2.contourArea(c) > self.min_area,input_data.contours))
        image_data:ImageData = self.get_original_image()
        image = image_data.image
        for contour in filtered_contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imshow("image", image)
        return ContoursData(filtered_contours)