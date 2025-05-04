import math
import os
import cv2
import time

from structure.runners.local.ocr_predictor import OcrPredictor
from structure.runners.local.score_image_for_text_detection import deep_score_image_for_text_detection
from structure.runners.runner import Runner
from structure.utils.image_data import ImageData
from structure.utils.pandas_utils import contains_register, get_name_from_path, save_or_update_table



# todo: decidir porcentagem que ira ser vizualizada por ocr
#todo: melhor parametro de pieces
#todo: mexer no parametro de resize
# todo: testar pre-filter amanha
# todo: decidir se redimensiona
# todo: analisar 2, 4, 5

class LocalRunner(Runner):
    def __init__(self, config):
        super().__init__(config)
        self.base_path = 'res/videos'
        # self.contours_crops = []


    def process(self,cap, start, end, pieces):
        max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("max frames:", max_frames)
        pieces_percent = (max_frames / pieces)
        start_frame = int(start * pieces_percent)
        end_frame = int(end * pieces_percent)
        print("start frame:", start_frame)
        print("end frame:", end_frame)


        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        contours_crops = []
        start_time = time.time()
        while cap.get(cv2.CAP_PROP_POS_FRAMES) != end_frame:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            # frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
            image_data = ImageData.from_image(frame)
            processed_data = self.pipeline.run(image_data)
            contours = processed_data.contours
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                crop = frame[y:y + h, x:x + w]
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

                # if prefilter_image(crop):  # <--- AQUI É O FILTRO!
                #     contours_crops.append(crop)
                contours_crops.append(crop)

        contours_sorted = [(crop, deep_score_image_for_text_detection(crop)) for crop in contours_crops]
        contours_sorted.sort(key=lambda x: x[1], reverse=True)
        ocr_predictor = OcrPredictor(ocr_options=[6])
        # todo: AJUSTE O NUMERO DE CONTORNOS
        result = ocr_predictor.perform_predictions(contours_sorted)
        print("result:", result)
        end_time = time.time()
        print("time: ", end_time - start_time)
        print(len(contours_sorted))

        # for cnt, r in contours_sorted:
        #     cv2.imshow('crop', cnt)
        #     cv2.waitKey(0)
        print(len(contours_crops))
        return len(result) > 0


    def run_pipeline(self, cap, pieces=2):
        start_time = time.time()
        end = math.ceil(pieces / 2)
        start = end - 1
        success = self.process(cap,start, end, pieces)
        previous_start = start

        next_end = end
        if not success:
            while not success:
                if previous_start > 0:
                    previous_start = previous_start - 1
                    previous_end = previous_start + 1
                    success = self.process(cap,previous_start , previous_end, pieces)
                elif not success and next_end < pieces:
                    next_end = next_end + 1
                    next_start = next_end - 1
                    success = self.process(cap, next_start, next_end, pieces)
                else:
                    break
        print("--- %s seconds ---" % (time.time() - start_time))




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
        video_paths = self.load_videos_path()
        video_path = video_paths[6]
        print(video_path)
        cap = cv2.VideoCapture(video_path)  # Assume o mesmo vídeo de teste
        self.run_pipeline(cap, pieces=31)



# 6 bem sucesso com filtro
# 7 bem sucesso com filtro
# 8 bem sucesso com filtro
# 9 analise
# 10 parcial sucesso com filtro
# 11 analise
# 12 bem sucesso com filtro
# 13 bem sucesso com filtro
# 13 bem sucesso com filtro com ressalvar (vale analise)
# 14 bem sucesso com filtro
# 15 sem sucesso com filtro mas sucesso sem filtro otimo para ajuste!!!!!
# 16 bem sucesso com filtro
# 17 bem sucesso com filtro com ressalvar (vale analise)
# 18 bem sucesso com filtro
# 19 bem sucesso com filtro
# 20 bem sucesso com filtro
# 21 bem sucesso com filtro com ressalvar (vale analise)
# 22 bem sucesso com filtro com ressalvar (vale analise)
# 22 bem sucesso com filtro com ressalvar (vale analise)
# 23 bem sucesso com filtro com ressalvar (vale analise)
# 24 nao sucessido nem sem filtro (principal problema e o pre-processamento)
# 25 sem sucesso com filtro mas sucesso sem filtro otimo para ajuste!!!!!
# 26 bem sucesso com filtro
# 27 bem sucesso com filtro
# 28 bem sucesso com filtro
# 29 bem sucesso com filtro
