from enum import Enum

from app.cv.black_filter import BlackFilter
from app.cv.cropper import Cropper
from app.cv.filter_by_area import FilterByArea
from app.cv.filter_mask_by_area import FilterMaskByArea
from app.cv.find_contours import FindContours
from app.ocr.ocr_pass import OcrPass
from structure.pipeline import Pipeline


class PipelineVariation(Enum):
    PIPELINE_1 = 1
    PIPELINE_OCR_5 = 2
    PIPELINE_OCR_6 = 3

class PipelineFactory:
    @staticmethod
    def create_pipeline(pipeline_variation: PipelineVariation) -> Pipeline:
        match pipeline_variation:
            case(PipelineVariation.PIPELINE_1):
                return Pipeline().add_passes(Cropper(0.8), BlackFilter, FilterMaskByArea, FindContours, FilterByArea, OcrPass)
            case PipelineVariation.PIPELINE_OCR_5:
                return Pipeline().add_passes(Cropper(0.8), BlackFilter, FilterMaskByArea, FindContours, FilterByArea, OcrPass(ocr_options=[5]))
            case PipelineVariation.PIPELINE_OCR_6:
                return Pipeline().add_passes(Cropper(0.8), BlackFilter, FilterMaskByArea, FindContours, FilterByArea, OcrPass(ocr_options=[6]))
