from enum import Enum

from structure.cv.black_filter import BlackFilter
from structure.cv.cropper import Cropper
from structure.cv.filter_by_area import FilterByArea
from structure.cv.filter_mask_by_area import FilterMaskByArea
from structure.cv.find_contours import FindContours
from structure.ocr.ocr_pass import OcrPass
from structure.pipeline import Pipeline


class PipelineVariation(Enum):
    NULL_PIPELINE = '0'
    PIPELINE_1 = '1'
    PIPELINE_OCR_5 = '2'
    PIPELINE_OCR_6 = '3'
    PIPELINE_OCR_12 = '4'
    PIPELINE_CONTOURS = '5'

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
            case PipelineVariation.PIPELINE_OCR_12:
                return Pipeline().add_passes(Cropper(0.8), BlackFilter, FilterMaskByArea, FindContours, FilterByArea, OcrPass(ocr_options=[12]))
            case PipelineVariation.NULL_PIPELINE, _:
                return Pipeline()
            case PipelineVariation.PIPELINE_CONTOURS:
                return Pipeline().add_passes(BlackFilter, FilterMaskByArea, FindContours, FilterByArea)
