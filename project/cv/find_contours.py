from project.pipeline import DetectionPass, I, O, IOComponent


class ContourData(IOComponent):
    def __init__(self, detected_contour):
        self.detected_contour = detected_contour


class FindContours(DetectionPass):
    def run(self, input_data: I) -> O:
        pass