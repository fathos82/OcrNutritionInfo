from time import time

import cv2
from websockets.sync.connection import Connection

from app.ocr.ocr_pass import OcrData
from project.cv_utils import get_time_code, resize_image
from project.pandas_utils import get_name_from_path, get_result_from_excel, save_or_update_table
from project.pipeline import Pipeline
from project.utils.image_data import ImageData
import pandas as pd






def run(**kwargs):
    video_path = kwargs['video_path']
    pipeline: Pipeline = kwargs['pipeline']
    test_name = kwargs.get('test_name', None)
    socket:Connection = kwargs.get('socket', None)

    if test_name is None:
        test_name = get_name_from_path(video_path)

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
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print('Video not found.')
            break
        # i += 1
        # if i % skip_frames == 0:
        #     continue
        if socket is not None:
            socket.ping()
        # TODO: Padronize tamanho com -> resize_image(frame)
        result: OcrData =  pipeline.run(ImageData.from_image(frame))

        result_set.update(result.word_set)
        count_contours += result.len_contours
        options.update(result.options)
        area_contours.extend(result.area_contours)

        if len(result_set) > 0 and time_to_find is None:
            time_to_find = get_time_code(cap.get(cv2.CAP_PROP_POS_MSEC))
            print("Time to find:", time_to_find)
            break
    end = time()

    # Carregando o Excel com os resultados esperados
    df = pd.read_excel("res/Planilha_Videos_Resultados.xlsx")

    expected_results = set(get_result_from_excel(df, test_name))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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


    return new_data
