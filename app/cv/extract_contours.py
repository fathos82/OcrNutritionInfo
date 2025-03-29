import cv2

from project.cv.find_contours import ContoursData
from project.pipeline import DetectionPass, I, O, IOComponent


class PossibilitiesData(IOComponent):
    def __init__(self, possibilities):
        super().__init__()
        self.possibilities:list = possibilities

class ExtractPossibilities(DetectionPass):

    def __init__(self, algorithm_name=''):
        super().__init__()
        self.algorithm_name = algorithm_name
    def run(self, input_data: ContoursData) -> O:
        possibilities = []
        for contour in input_data.contours:
            area = cv2.contourArea(contour)
            possibility = self.get_original_image()
            contour = {
                'area': area,
                'possibilities': contour,
            }
            possibilities.append(contour)

        return PossibilitiesData(possibilities)





