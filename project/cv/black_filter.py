import cv2
import numpy as np
from line_profiler_pycharm import profile



import cv2
import numpy as np
from project.pipeline import DetectionPass
from project.utils.image_data import ImageData


class BlackFilter(DetectionPass):
    @profile
    def run(self, input_data: ImageData) -> ImageData:
        img = input_data.image

        # Certifique-se de que a imagem está em uint8 para evitar conversões desnecessárias
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)

        # Calcular o canal K diretamente sem conversões extras
        kChannel = 255 - np.max(img, axis=2)  # Forma otimizada

        # cv2.imshow("Black Filter", kChannel)

        # Aplicar threshold binário diretamente
        _, binaryImage = cv2.threshold(kChannel, 160, 255, cv2.THRESH_BINARY)

        return ImageData.from_image(binaryImage)
