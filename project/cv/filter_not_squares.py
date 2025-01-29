import cv2

from project.cv.find_contours import ContoursData
from project.pipeline import DetectionPass, I, O



class FilterNotSquares(DetectionPass):
    def run(self, input_data: ContoursData) -> ContoursData:
        temporary_image = input_data.temporary_image
        contours = input_data.contours
        filtered_contours = []
        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) == 4:
                filtered_contours.append(contour)
        return ContoursData(filtered_contours, temporary_image)