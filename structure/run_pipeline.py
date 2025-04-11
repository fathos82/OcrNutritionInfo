import multiprocessing
import os.path
from time import time
from typing import cast

import cv2
from websockets.sync.connection import Connection

from structure.ocr.ocr_pass import OcrData
from structure.pipeline import Pipeline
from structure.utils.cv_utils import get_time_code
from structure.utils.image_data import ImageData
import pandas as pd

from structure.utils.pandas_utils import get_name_from_path, get_result_from_excel


def run( **kwargs):
    video_path = kwargs['video_path']
    pipeline: Pipeline = kwargs['pipeline']
    test_name = kwargs.get('test_name', None)
    queue:multiprocessing.Queue= kwargs['queue']
    print(test_name)

    if test_name is None:
        test_name = get_name_from_path(video_path)
    else:
        test_name = get_name_from_path(test_name)

    skip_frames = 2

    # pipeline = Pipeline().add_passes(Cropper(0.8), BlackFilter, FilterMaskByArea, FindContours, FilterByArea, OcrPass)
    cap = cv2.VideoCapture(video_path)
    result_set = set()
    options = set()
    count_contours = 0
    time_to_find = None
    area_contours = []
    start_time = time()
    # Processando o vídeo
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) + 1
    print(frame_count)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print('Video not found.')
            break
        # i += 1
        # if i % skip_frames == 0:
        #     continue

        # TODO: Padronize tamanho com -> resize_image(frame)
        result: OcrData =  cast(OcrData, pipeline.run(ImageData.from_image(frame)))

        result_set.update(result.word_set)
        count_contours += result.len_contours
        options.update(result.options)
        area_contours.extend(result.area_contours)

        if len(result_set) > 0 and time_to_find is None:
            print("time to find")
            time_to_find = get_time_code(cap.get(cv2.CAP_PROP_POS_MSEC))
            print("Time to find:", time_to_find)
            # break
    end = time()
    print("Elapsed time:", end - start_time)

    # Carregando o Excel com os resultados esperados
    df = pd.read_excel("res/Planilha_Videos_Resultados.xlsx")

    expected_results = set(get_result_from_excel(df, test_name))
    # Calculando os resultados
    process_time = end - start_time
    median_contour = count_contours // frame_count
    success = result_set == expected_results

    # Criando um novo DataFrame com os resultados
    new_data = {
        "Id": [test_name],
        "Resultado Obtido": [", ".join(sorted(result_set))],
        "Resultado Esperado": [", ".join(sorted(expected_results))],
        "Sucedido": [success],  # Usando 'FALSO' ou 'VERDADEIRO'
        "Momento Exato Da Primeira Inferencia Sucedida": [time_to_find],
        "OCR Options": [", ".join(map(str, sorted(options)))],
        "Tempo Processamento": [process_time],
        "Área dos Contornos": [", ".join(map(str, area_contours))],
        "Qntd Média de Contornos": [median_contour]
    }
    print(new_data)
    os.remove(video_path)
    queue.put(new_data)

