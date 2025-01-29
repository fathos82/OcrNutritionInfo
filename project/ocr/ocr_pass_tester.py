import re
from time import time

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
        return str(self.word_set)





class OcrPassTester(DetectionPass):
    def __init__(self):
        super().__init__()
        self.options = ["SODIO", "AÇUCAR ADICIONADO", "GORDURA SATURADA"]

    def filter_right_words(self, words, confidence_threshold=0.6):
        words_set = set()
        for word in words:
            cleaned_word = re.sub(r'[^a-zA-Z0-9\s]', '', word)
            cleaned_word = cleaned_word.replace('0', 'O').replace('1', 'I')
            word_upper = cleaned_word.upper()
            match = difflib.get_close_matches(word_upper, self.options, cutoff=confidence_threshold)
            words_set.update(match)
        return words_set
    def run(self, start_input:ImageData) -> OcrData:
        # TODO: Pre PROCESSING ANG GET CONTOURS
        words_set = set()
        image = start_input.image

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0) #TODO: --> POSSIVEL NECESSIDADE DE AJUSTES
        otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)[1]
        inv = cv2.bitwise_not(otsu)
        transformed = self.thick(otsu)



        for i in range(3,13):
            start = time()
            custom_config = '--psm '+str(i)
            print(custom_config)
            text = pytesseract.image_to_string(transformed, config=custom_config)
            local_words_set = self.filter_right_words(text.split())
            print(local_words_set)
            words_set.update(local_words_set)
        cv2.imshow('OCR', transformed)
        cv2.imshow('otsu', otsu)
        end = time()
        # print("Time Taken:", end-start)

        return OcrData(words_set)
    def thick(self, image):
        negated = cv2.bitwise_not(image)
        kernel = np.ones((2,2), np.uint8)
        transformed = cv2.dilate(negated, kernel, iterations=1)
        image = cv2.bitwise_not(transformed)
        return  image

    def thin(self, image):
        negated = cv2.bitwise_not(image)
        kernel = np.ones((2,2), np.uint8)
        transformed = cv2.erode(negated, kernel, iterations=1)
        image = cv2.bitwise_not(transformed)
        return image