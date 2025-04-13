import os
import cv2
import numpy as np
from structure.cv.find_contours import ContoursData
from structure.pipeline_factory import PipelineFactory, PipelineVariation
from structure.run_pipeline import run
from structure.runners.runner import Runner
from structure.utils.image_data import ImageData
from structure.utils.pandas_utils import contains_register, get_name_from_path, save_or_update_table


class LocalRunner(Runner):
    def __init__(self, config):
        super().__init__(config)
        self.base_path = 'res/videos'
        self.contours_crops = []  # Armazena tuplas (recorte, média_de_luminância, área)

    def load_videos_path(self):
        video_paths = []
        paths = os.listdir(self.base_path)
        for video_file in paths:
            if not video_file.endswith('.mp4') or contains_register(get_name_from_path(video_file)):
                continue
            video_path = os.path.join(self.base_path, video_file)
            video_paths.append(video_path)
        return video_paths

    def run(self):
        load_videos_path = self.load_videos_path()
        videos_path = load_videos_path[6]
        cap = cv2.VideoCapture(videos_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            # Verifica se o frame está colorido (3 canais) ou em grayscale (1 canal)
            if len(frame.shape) == 2:  # Se estiver em grayscale
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # Converte para BGR

            # Processa o frame e obtém os contornos
            image_data = ImageData.from_image(frame)
            processed_data = self.pipeline.run(image_data)
            contours = processed_data.contours  # Note o nome correto da variável

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                crop = frame[y:y + h, x:x + w]  # Recorte colorido (BGR)

                # Calcula a média de luminância (converte apenas para cálculo)
                gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                mean_luminance = np.mean(gray_crop)

                # Calcula a área do contorno
                area = cv2.contourArea(contour)

                # Armazena o recorte colorido, média e área
                self.contours_crops.append((crop, mean_luminance, area))  # Note o nome correto

        # Ordenação
        self.contours_crops.sort(key=lambda x: (x[1], -x[2]))

        # Exibe os recortes coloridos ordenados
        for idx, (crop, mean, area) in enumerate((self.contours_crops)):
            print(f"Recorte : {area}")
            print(f"Crop : {mean}")

            cv2.imshow(f"Recorte ", crop)
            cv2.waitKey(0)

        cap.release()
        cv2.destroyAllWindows()



#240
#550
#720
#1300
#390
#3400




















