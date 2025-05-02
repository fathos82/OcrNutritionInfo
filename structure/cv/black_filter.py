import cv2
import numpy as np



import cv2
import numpy as np
from structure.pipeline import DetectionPass
from structure.utils.image_data import ImageData


class BlackFilter(DetectionPass):
    def run(self, input_data: ImageData) -> ImageData:
        img = input_data.image

        # Certifique-se de que a imagem está em uint8 para evitar conversões desnecessárias
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)

        # Calcular o canal K diretamente sem conversões extras
        kChannel = 255 - np.max(img, axis=2)  # Forma otimizada
        # cv2.imshow('kChannel', kChannel)

        # cv2.imshow("Black Filter", kChannel)

        # Aplicar threshold binário diretamente
        _, binaryImage = cv2.threshold(kChannel, 160, 255, cv2.THRESH_BINARY)
        # cv2.imshow("Black Filter", binaryImage)
        # cv2.waitKey(0)

        return ImageData.from_image(binaryImage)
