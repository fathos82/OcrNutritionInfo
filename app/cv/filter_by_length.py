import cv2

from app.cv.find_contours import ContoursData
from structure.pipeline import DetectionPass, O
from structure.utils.image_data import ImageData


class FilterByLength(DetectionPass):
    def __init__(self, min_length=300, max_length=1000):
        super().__init__()
        self.min_length = min_length
        self.max_length = max_length

    def run(self, input_data: ContoursData) -> ContoursData:
        import cv2


        filtered_contours = [contour for contour in input_data.contours if
                             self.min_length < cv2.arcLength(contour, True) < self.max_length]

        # print(len(filtered_contours))
        # image = self.get_original_image()
        # for contour in filtered_contours:
        #     x, y, w, h = cv2.boundingRect(contour)
        #     cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # cv2.imshow("image", image)

        return ContoursData(filtered_contours)