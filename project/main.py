# TODO: Pesquisar flake8, black, sphinx
from time import time

import cv2

from project.cv.filter_by_area import FilterByArea
from project.cv.filter_not_squares import FilterNotSquares
from project.cv.find_contours import FindContours
from project.cv.preprocess_pass import PreProcessPass
from project.ocr.ocr_pass import OcrPass, OcrData
from project.ocr.ocr_pass_tester import OcrPassTester
from project.pipeline import Pipeline, DetectionPass
from project.utils.image_data import ImageData


#TODO: MultThreading Partion,
# Change Lib To Process Video,
# IMPROVE FILTERING,
# RESIZE IMAGEM, CROP
# IMAGE (CENTER)
# APLYING N TESSERECAT CONFIG


def run_from_video(id):

    video_config_set = [
        {
            "name_file": "1.mp4",
            "ocr_options": [5]
        },
        {
            "name_file": "2.mp4",
            "ocr_options": [5]
        },
        {
            "name_file": "3.mp4",
            "ocr_options": [12]
        },
        {
            "name_file": "4.mp4",
            "ocr_options": [6,12]
        },
    ]

    video_config = video_config_set[id]
    # MY VERSION:
    pipeline = Pipeline().add_passes(PreProcessPass, FindContours,  FilterNotSquares, OcrPass(ocr_options=video_config["ocr_options"])) # TODO: CRIAR PASSO RESIZE


    cap = cv2.VideoCapture(f"project/res/videos/{video_config['name_file']}")
    ret, frame = cap.read()
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    data: OcrData = pipeline.run(ImageData.from_image(frame))
    word_set = set()
    start_total = time()

    skip_frames = 2
    counter = 0

    try:
        while cap.isOpened():
            if not ret:
                print("Can't read frame")
                break
            ret, frame = cap.read()
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            counter+=1
            if counter % skip_frames == 0:
                data: OcrData = pipeline.run(ImageData.from_image(frame))
                if len(word_set) > 0:
                    raise Exception()
                end = time()
            word_set.update(data.word_set)

    except Exception as e:
        end_total = time()
        print("Total time:", end_total - start_total)
        print(word_set)
    finally:
        print(word_set)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def run_from_image(id):
    image_config_set = [

        {
            "name_file": "2.jpg",
            "ocr_options": [6, 12]
        },
        {
            "name_file": "3.jpeg",
            "ocr_options": [6, 12]
        }
    ]



# pipeline = Pipeline().add_passes(PreProcessPass, FindContours, FilterByArea(max_area=1500), OcrPass(ocr_options=[5,6,12]))
#
# image = cv2.imread("project/res/images/1.jpg")
# # image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
# data:OcrData = pipeline.run(ImageData.from_image(image))
# print(data.word_set)
# cv2.waitKey(0)


run_from_video(3)


# run_from_image()








pipeline = Pipeline().add_passes(PreProcessPass, FindContours, FilterByArea(min_area=300, max_area=6000)) # TODO: CRIAR PASSO RESIZE



cap = cv2.VideoCapture(f"project/res/videos/3.mp4")
ret, frame = cap.read()
frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
data: OcrData = pipeline.run(ImageData.from_image(frame))
skip_frames = 2
counter = 0

try:
    while cap.isOpened():
        if not ret:
            print("Can't read frame")
            break
        ret, frame = cap.read()
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        counter+=1
        if counter % skip_frames == 0:
            data = pipeline.run(ImageData.from_image(frame))
        cv2.waitKey(0)

except Exception as e:
    end_total = time()
finally:

    cv2.destroyAllWindows()



#
# import cv2
# import threading
# from time import time
# import numpy as np
#
# from project.cv.filter_by_area import FilterByArea
# from project.cv.find_contours import FindContours
# from project.cv.preprocess_pass import PreProcessPass
# from project.ocr.ocr_pass import OcrPass, OcrData
# from project.pipeline import Pipeline
# from project.utils.image_data import ImageData
#
# # Configuração do pipeline
# pipeline = Pipeline().add_passes(PreProcessPass, FindContours, FilterByArea, OcrPass(ocr_options=[6,12]))
#
# video_path = "project/res/videos/3.mp4"
# num_threads = 2 # Quantidade de threads/pedaços do vídeo
#
# # Abrindo o vídeo para obter informações
# cap = cv2.VideoCapture(video_path)
# fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames por segundo
# total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total de frames
# duration = total_frames / fps  # Duração em segundos
# cap.release()
#
# # Calculando os pedaços do vídeo
# chunk_size = total_frames // num_threads  # Quantidade de frames por thread
#
# word_set = set()
# start_total = time()
# lock = threading.Lock()  # Para evitar concorrência ao acessar word_set
#
#
# # ✅ Função que cada thread executará
# def process_video_chunk(start_frame, end_frame):
#     cap = cv2.VideoCapture(video_path)
#     cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # Vai para o frame inicial
#
#     while cap.isOpened():
#         frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
#         if frame_number >= end_frame:
#             break  # Sai se chegou ao final do segmento
#
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         # Redimensionamento para acelerar o processamento
#         frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
#
#         # Processa o frame com o pipeline
#         data: OcrData = pipeline.run(ImageData.from_image(frame))
#
#         # Atualiza word_set de forma segura
#         with lock:
#             word_set.update(data.word_set)
#
#         # Exibe o frame processado (opcional)
#         # cv2.imshow(f"Thread {start_frame}-{end_frame}", frame)
#         # if cv2.waitKey(1) & 0xFF == ord('q'):
#         #     break
#
#     cap.release()
#
#
# # Criando e iniciando as threads
# threads = []
# for i in range(num_threads):
#     start_frame = i * chunk_size
#     end_frame = (i + 1) * chunk_size if i < num_threads - 1 else total_frames
#     thread = threading.Thread(target=process_video_chunk, args=(start_frame, end_frame))
#     threads.append(thread)
#     thread.start()
#
# # Aguardando todas as threads finalizarem
# for thread in threads:
#     thread.join()
#
# cv2.destroyAllWindows()
#
# end_total = time()
# print("Total time:", end_total - start_total)
# print(word_set)
#
#
#
#
#



# THREAD TO CAP VERSION
# from time import time
# import cv2
# import threading
# import queue
#
# from project.cv.filter_by_area import FilterByArea
# from project.cv.find_contours import FindContours
# from project.cv.preprocess_pass import PreProcessPass
# from project.ocr.ocr_pass import OcrPass, OcrData
# from project.pipeline import Pipeline, DetectionPass
# from project.utils.image_data import ImageData
#
# # Configuração do pipeline
# pipeline = Pipeline().add_passes(PreProcessPass, FindContours, FilterByArea, OcrPass(ocr_options=[5,6,12]))
#
# # Configuração do vídeo
# video_path = "project/res/videos/2.mp4"
# frame_skip = 2  # Pular frames para otimização
# frame_queue = queue.Queue(maxsize=10)  # Fila para armazenar frames
#
# word_set = set()
# start_total = time()
#
#
# # ✅ Thread para captura de frames
# def capture_frames(video_path):
#     cap = cv2.VideoCapture(video_path)
#     frame_count = 0
#
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         # Redimensiona antes de enviar para processamento
#         frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
#
#         # Pula frames desnecessários
#         if frame_count % frame_skip == 0:
#             frame_queue.put(frame)
#
#         frame_count += 1
#
#     cap.release()
#     frame_queue.put(None)  # Sinaliza fim do vídeo
#
#
# # ✅ Thread para processamento de frames
# def process_frames():
#     while True:
#         frame = frame_queue.get()  # Pega um frame da fila
#         if frame is None:
#             break  # Sai do loop se o vídeo terminou
#
#         start = time()
#         data: OcrData = pipeline.run(ImageData.from_image(frame))
#         end = time()
#
#         word_set.update(data.word_set)
#
#         # Exibir frame processado (opcional)
#
#
# # Criando as threads
# capture_thread = threading.Thread(target=capture_frames, args=(video_path,))
# process_thread = threading.Thread(target=process_frames)
#
# # Iniciando as threads
# capture_thread.start()
# process_thread.start()
#
# # Esperando as threads terminarem
# capture_thread.join()
# process_thread.join()
#
# cv2.destroyAllWindows()
#
# end_total = time()
# print("Total time:", end_total - start_total)
# print(word_set)















































