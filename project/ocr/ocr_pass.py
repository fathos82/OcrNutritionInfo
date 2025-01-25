from project.pipeline import DetectionPass
from project.pipeline import IOComponent


# TODO: Temporary Constructor
class OcrData(IOComponent):
    pass




class OcrPass(DetectionPass):
    def run(self, start_input) -> OcrData:
        print("Ocr Pass")
        return OcrData()