from time import time

import cv2
import numpy as np

from project.cv.black_filter import BlackFilter
from project.cv.filter_by_area import FilterByArea
from project.cv.filter_by_length import FilterByLength
from project.cv.filter_mask_by_area import FilterMaskByArea
from project.cv.find_contours import FindContours
from project.cv.preprocess_pass import PreProcessPass
from project.cv.white_filter import WhiteFilter
from project.ocr.ocr_pass import OcrPass, OcrData
from project.ocr.ocr_pass_tester import OcrPassTester
from project.pipeline import Pipeline
from project.utils.image_data import ImageData


# CRIAR UM RUNNER
# DEIXAR PARAMTROS IMPORTANTES PERSONALIZAVVEIS
# AJUSTAR PARAMETROS
# PADRONIZAR ENTRADA
# FINALIZAR ESTRTURA DE PIPELINE
# CRIAR ALGORITIMO IMPIRICO PARA TENTAR OTIMIZAR A RESPOSTAR

def resize_image(image, new_width=720):
    height, width = image.shape[:2]
    new_height = int(height * new_width / width)
    new_image = cv2.resize(image, (new_width, new_height))
    return new_image


pipeline = Pipeline().add_passes(BlackFilter, FindContours, FilterByArea)  # TODO: CRIAR PASSO RESIZE
cap = cv2.VideoCapture('project/res/videos/Vídeo 4.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = resize_image(frame)
    pipeline.run(ImageData.from_image(frame))
    cv2.imshow('frame', frame)
    cv2.waitKey(0)

































