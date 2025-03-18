import cv2
import numpy as np

from project.pipeline import DetectionPass
from project.utils.image_data import ImageData


class BlackFilter(DetectionPass):
    def run(self, input_data: ImageData) -> ImageData:
        img = input_data.image

        # Certifique-se de que a imagem está no formato correto
        img = img.astype(np.float32) / 255.0  # Usa np.float32 em vez de np.float

        # Calcular o canal K corretamente
        kChannel = 1 - np.max(img, axis=2)

        # Converter para uint8 corretamente
        kChannel = (kChannel * 255).astype(np.uint8)
        binaryThresh = 160
        _, binaryImage = cv2.threshold(kChannel, binaryThresh, 255, cv2.THRESH_BINARY)
        cv2.imshow('kChannel', kChannel)
        cv2.imshow('binaryImage', binaryImage)
        return ImageData.from_image(kChannel)
