from time import time

import cv2
import numpy as np
from project.cv.filter_by_area import FilterByArea
from project.cv.filter_by_length import FilterByLength
from project.cv.find_contours import FindContours
from project.cv.preprocess_pass import PreProcessPass
from project.ocr.ocr_pass import OcrPass, OcrData
from project.pipeline import Pipeline
from project.utils.image_data import ImageData


# CRIAR UM RUNNER
# DEIXAR PARAMTROS IMPORTANTES PERSONALIZAVVEIS
# AJUSTAR PARAMETROS
# PADRONIZAR ENTRADA
# FINALIZAR ESTRTURA DE PIPELINE
# CRIAR ALGORITIMO IMPIRICO PARA TENTAR OTIMIZAR A RESPOSTAR


pipeline = Pipeline().add_passes(PreProcessPass, FindContours, FilterByLength)  # TODO: CRIAR PASSO RESIZE

cap = cv2.VideoCapture(f"project/res/videos/1.mp4")
ret, frame = cap.read()
frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
data = pipeline.run(ImageData.from_image(frame))
word_set = set()

skip_frames = 2
counter = 0
start = time()


try:
    while cap.isOpened():
        if not ret:
            print("Can't read frame")
            break
        ret, frame = cap.read()
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        counter += 1
        if counter % skip_frames == 0:
            data: OcrData = pipeline.run(ImageData.from_image(frame))
            # if len(word_set) > 0:
            #     raise Exception()
        # cv2.waitKey(0)

        # word_set.update(data.word_set)
except Exception as e:
    print(e)

finally:
    # print("ALTO EM: " + str(word_set))
    end = time()
    print("Total time: " + str(end - start))
    cv2.waitKey(0)
    cv2.destroyAllWindows()





























