import random
import re
from time import time
from typing import List
import cv2
import numpy as np
import pytesseract
import difflib
import multiprocessing as mp

class OcrPredictor:
    def __init__(self, ocr_options: List[int] = [6, 12,5], number_cores=mp.cpu_count()//2):
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

    def process_contour(self, contour, score):
        # x, y, w, h = cv2.boundingRect(contour)
        # crop_image = image[y:y + h, x:x + w]
        # gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(contour, (3, 3), 0)  # TODO: Possível necessidade de ajustes
        otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)[1]
        successful_opts = set()
        words_set = set()

        for opt in self.ocr_options:
            custom_config = f'--psm {opt}'
            text = pytesseract.image_to_string(otsu, config=custom_config) # TODO: ADICIONAR LINGUA PORTUGUESA
            word = self.filter_right_words(text.split())
            words_set.update(word)
            if len(words_set) > 0:
                successful_opts.add(opt)
                break
        return words_set
    def perform_predictions(self, contours):
        # contours = start_input.contours
        # for contour in contours:
        #     words_set = self.process_contour(contour, image)

        with mp.Pool(processes=self.number_cores) as pool: # TODO: EVITAR ITERAR SOBRE VARIAS CONTORNOS
            # TODO: CRIAR UMA ESTRATEGIA, CASO ENCONTRADO ITERAR APENAS DOS CONTORNOS PROXIMOS!
            results = pool.starmap(self.process_contour, [ (contour, score) for contour, score in contours])
        # Unir os resultados de todos os processos
        words_set = set()
        for result in results:
            words_set.update(result)
        return words_set

