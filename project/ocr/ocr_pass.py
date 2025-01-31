import re
from time import time
from typing import List

import cv2
import numpy as np
import pytesseract

from project.cv.find_contours import ContoursData
from project.pipeline import DetectionPass
from project.pipeline import IOComponent
from project.utils.image_data import ImageData
import difflib


# TODO: Temporary Constructor
class OcrData(IOComponent):
    def __init__(self, word_set):
        self.word_set = word_set
    def __str__(self):
        return "OcrData(word_set={})".format(self.word_set)


class OcrPass(DetectionPass):
    def __init__(self, temporary_image_data: ImageData, ocr_options: List[int] = [6, 12]):
        super().__init__()
        self.temporary_image_data = temporary_image_data
        self.options = ["SODIO", "AÇUCAR ADICIONADO", "GORDURA SATURADA"]
        self.ocr_options = ocr_options

    def filter_right_words(self, words, confidence_threshold=0.6):
        words_set = set()
        for word in words:
            cleaned_word = re.sub(r'[^a-zA-Z0-9\s]', '', word)
            cleaned_word = cleaned_word.replace('0', 'O').replace('1', 'I')
            word_upper = cleaned_word.upper()
            match = difflib.get_close_matches(word_upper, self.options, cutoff=confidence_threshold)
            words_set.update(match)
        return words_set

    def run(self, start_input: ContoursData) -> OcrData:
        words_set = set()
        # image = self.get_original_image()
        image = self.temporary_image_data.image
        for contour in start_input.contours:
            x, y, w, h = cv2.boundingRect(contour)
            crop_image = image[y:y + h, x:x + w]
            gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (3, 3), 0)  #TODO: --> POSSIVEL NECESSIDADE DE AJUSTES
            otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)[1]
            transformed = self.thick(otsu)

            for opt in self.ocr_options:
                start = time()
                custom_config = f'--psm {opt}'
                # print(custom_config)
                text = pytesseract.image_to_string(transformed, config=custom_config)
                words_set.update(self.filter_right_words(text.split()))
            end = time()
            # print("Time Taken:", end-start)

        return OcrData(words_set)

    def thick(self, image):
        negated = cv2.bitwise_not(image)
        kernel = np.ones((1, 1), np.uint8)
        transformed = cv2.dilate(negated, kernel, iterations=1)
        image = cv2.bitwise_not(transformed)
        return image

    def thin(self, image):
        negated = cv2.bitwise_not(image)
        kernel = np.ones((2, 2), np.uint8)
        transformed = cv2.erode(negated, kernel, iterations=1)
        image = cv2.bitwise_not(transformed)
        return image
