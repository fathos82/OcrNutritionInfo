from project.cv.processed_image import ProcessedImage
from project.ocr.ocr_data import OcrData
from project.pipeline import DetectionPass



class OcrPass(DetectionPass):
    def run(self, start_input:ProcessedImage) -> OcrData:
        print("Ocr Pass")
        return OcrData()