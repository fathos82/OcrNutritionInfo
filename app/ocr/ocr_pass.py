import random
import re
from time import time
from typing import List
import cv2
import numpy as np
import pytesseract
import difflib
import multiprocessing as mp


from app.cv.find_contours import ContoursData
from project.pipeline import DetectionPass
from project.pipeline import IOComponent


# TODO: Temporary Constructor
class OcrData(IOComponent):
    def __init__(self, word_set, options, len_contours, area_contours):
        self.options = options
        self.word_set = word_set
        self.len_contours = len_contours
        self.area_contours  = area_contours

    def __str__(self):
        return "OcrData(word_set={})".format(self.word_set)





class OcrPass(DetectionPass):
    def __init__(self, ocr_options: List[int] = [6, 12,5], number_cores=mp.cpu_count()//2):
        super().__init__()
        self.options = ["SODIO", "SÓDIO","AÇUCAR ADICIONADO", "GORDURA SATURADA", "AÇUCAR", "AGUCAR", "SATURADA", "GORDURA"]
        self.ocr_options = ocr_options
        self.number_cores = number_cores

    def apply_corrections(self, match):

        corrections = {
            frozenset(["AÇUCAR", "AGUCAR", "ADICIONADO, AÇUCAR ADICIONADO"]): "AÇUCAR ADICIONADO",
            frozenset(["SATURADA", "GORDURA"]): "GORDURA SATURADA",
            frozenset(["SÓDIO"]): "SODIO"
        }
        for k, v in corrections.items():
            for i in range(len(match)):
                if match[i] in k:

                    if v not in match:
                        match[i] = v
                    else:
                        match.pop(i)
        return match


    def filter_right_words(self, words, confidence_threshold=0.7):
        words_set = set()
        for word in words:
            cleaned_word = re.sub(r'[^a-zA-Z0-9\s]', '', word)
            cleaned_word = cleaned_word.replace('0', 'O').replace('1', 'I')
            word_upper = cleaned_word.upper()
            match = difflib.get_close_matches(word_upper, self.options, cutoff=confidence_threshold)
            match = self.apply_corrections(match)
            words_set.update(match)
        return words_set

    def process_contour(self, contour, image):
        x, y, w, h = cv2.boundingRect(contour)
        crop_image = image[y:y + h, x:x + w]
        gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)  # TODO: Possível necessidade de ajustes
        otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)[1]
        successful_opts = set()
        words_set = set()
        _contour = None

        for opt in self.ocr_options:
            custom_config = f'--psm {opt}'
            text = pytesseract.image_to_string(otsu, config=custom_config) # TODO: ADICIONAR LINGUA PORTUGUESA
            contour_area = None

            word = self.filter_right_words(text.split())
            words_set.update(word)
            if len(words_set) > 0:
                successful_opts.add(opt)
                contour_area = cv2.contourArea(contour)
                break
        return words_set, successful_opts, contour_area
    def run(self, start_input: ContoursData) -> OcrData:
        image = self.get_original_image()

        # contours = start_input.contours
        # for contour in contours:
        #     words_set = self.process_contour(contour, image)

        with mp.Pool(processes=self.number_cores) as pool: # TODO: EVITAR ITERAR SOBRE VARIAS CONTORNOS
            # TODO: CRIAR UMA ESTRATEGIA, CASO ENCONTRADO ITERAR APENAS DOS CONTORNOS PROXIMOS!
            results = pool.starmap(self.process_contour, [(contour, image) for contour in start_input.contours])
        # Unir os resultados de todos os processos
        words_set = set()
        successful_opts = set()
        area_contours = []
        contour_area = None
        for result in results:
            words_set.update(result[0])  # Unir as palavras
            successful_opts.update(result[1])  # Unir as opções bem-sucedidas
            contour_area = result[2]
            if contour_area is not None:
                area_contours.append(contour_area)

        return OcrData(words_set, successful_opts, len(start_input.contours), area_contours)

    def thick(self, image):
        negated = cv2.bitwise_not(image)
        kernel = np.ones((1, 1), np.uint8)
        transformed = cv2.dilate(negated, kernel, iterations=1)
        return cv2.bitwise_not(transformed)

    def thin(self, image):
        negated = cv2.bitwise_not(image)
        kernel = np.ones((2, 2), np.uint8)
        transformed = cv2.erode(negated, kernel, iterations=1)
        return cv2.bitwise_not(transformed)
