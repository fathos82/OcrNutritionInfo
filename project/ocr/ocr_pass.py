import re
from time import time
from typing import List
import cv2
import numpy as np
import pytesseract
import difflib
import multiprocessing as mp

from project.cv.find_contours import ContoursData
from project.pipeline import DetectionPass
from project.pipeline import IOComponent


# TODO: Temporary Constructor
class OcrData(IOComponent):
    def __init__(self, word_set):
        self.word_set = word_set

    def __str__(self):
        return "OcrData(word_set={})".format(self.word_set)


class OcrPass(DetectionPass):
    def __init__(self, ocr_options: List[int] = [6, 12], number_cores=mp.cpu_count()//2):
        super().__init__()
        self.options = ["SODIO", "AÇUCAR ADICIONADO", "GORDURA SATURADA"]
        self.ocr_options = ocr_options
        self.number_cores = number_cores

    def filter_right_words(self, words, confidence_threshold=0.6):
        words_set = set()
        for word in words:
            cleaned_word = re.sub(r'[^a-zA-Z0-9\s]', '', word)
            cleaned_word = cleaned_word.replace('0', 'O').replace('1', 'I')
            word_upper = cleaned_word.upper()
            match = difflib.get_close_matches(word_upper, self.options, cutoff=confidence_threshold)
            words_set.update(match)
        return words_set

    def process_contour(self, contour, image):
        x, y, w, h = cv2.boundingRect(contour)
        crop_image = image[y:y + h, x:x + w]
        gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)  # TODO: Possível necessidade de ajustes
        otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)[1]
        transformed = self.thick(otsu)

        words_set = set()
        for opt in self.ocr_options:
            custom_config = f'--psm {opt}'
            text = pytesseract.image_to_string(transformed, config=custom_config)
            words_set.update(self.filter_right_words(text.split()))
        return words_set

    def run(self, start_input: ContoursData) -> OcrData:
        image = self.get_original_image()

        with mp.Pool(processes=self.number_cores) as pool:
            results = pool.starmap(self.process_contour, [(contour, image) for contour in start_input.contours])

        # Unir os resultados de todos os processos
        words_set = set().union(*results)
        return OcrData(words_set)

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
