import os
import unicodedata
from os import waitid_result
from time import time, process_time

import cv2
from django.contrib.messages import success

from project.cv.black_filter import BlackFilter
from project.cv.cropper import Cropper
from project.cv.extract_contours import ExtractPossibilities, PossibilitiesData
from project.cv.filter_mask_by_area import FilterMaskByArea
from project.cv.find_contours import FindContours
from project.cv.filter_by_area import FilterByArea
from project.ocr.ocr_pass import OcrPass, OcrData
from project.pipeline import Pipeline
from project.utils.image_data import ImageData
import pandas as pd


# TODO: Heuristica que pega alguns frames apos a primeira inferencia bem sucedida, e depois tenta buscar a segunda

def resize_image(image, new_width=720):
    """Redimensiona a imagem mantendo a proporção."""
    height, width = image.shape[:2]
    new_height = int(height * new_width / width)
    return cv2.resize(image, (new_width, new_height))

def capture_frame(frame, time_code, file_path):
    file_name = file_path.split('/')[-1]
    file_name = file_name.split('.')[0]
    cv2.imwrite(f"project/res/frames/{file_name}_{time_code}.jpg", frame)

def get_time_code(time_ms):
    seconds = int(time_ms / 1000)
    minutes = int(seconds / 60)
    hours = int(minutes / 60)
    return f"{hours:02d}_{minutes%60:02d}_{seconds%60:02d}_{int(time_ms):02d}"

def get_name_from_path(video_path):
    name = video_path.split('/')[-1]
    name = name.split('.')[0]
    return name


def get_result_from_excel(df, name_video):
    result = df[df['Número do Vídeo'].str.strip() == name_video]['Resultado Esperado'].values
    if len(result) == 0:
        print(f"Erro: Nenhum resultado encontrado para o vídeo {name_video}")
        return []  # Retorna uma lista vazia caso não encontre o vídeo

    # Formata o resultado
    formated_result = str(result[0]).split(',')[-1].upper().strip()
    formated_result =  ''.join(
        c if c == 'ç' else c for c in unicodedata.normalize('NFD', formated_result)
        if unicodedata.category(c) != 'Mn' or c == 'ç'
    )

    return formated_result.split('e')


def save_or_update_table(new_data, file_path='./project/res/result/tabela_resultados.xlsx'):
    # Verificando se o arquivo já existe
    if os.path.exists(file_path):
        existing_df = pd.read_excel(file_path)  # Substituído para ler arquivo Excel
        updated_df = pd.concat([existing_df, new_data], ignore_index=True)
    else:
        updated_df = new_data

    updated_df.to_excel(file_path, index=False)
    print(updated_df)
    return file_path

def run(video_name):
    video_path = 'project/res/videos/'+ video_name +'.mp4'
    cap = cv2.VideoCapture(video_path)
    skip_frames = 2
    pipeline = Pipeline().add_passes(Cropper(0.8), BlackFilter, FilterMaskByArea, FindContours, FilterByArea, OcrPass)

    result_set = set()
    options = set()
    i = 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    count_contours = 0
    time_to_find = None
    area_contours = []
    start_time = time()
    name = get_name_from_path(video_path)
    df = pd.read_excel("project/res/Planilha_Videos_Resultados.xlsx")


    # Processando o vídeo
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print('Video not found.')
            break
        # i += 1
        # if i % skip_frames == 0:
        #     continue
        result: OcrData = pipeline.run(ImageData.from_image(resize_image(frame)))

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
    df = pd.read_excel("project/res/Planilha_Videos_Resultados.xlsx")
    name = get_name_from_path(video_path)
    print(name)
    expected_results = set(get_result_from_excel(df, name))

    # Calculando os resultados
    process_time = end - start_time
    median_contour = count_contours // frame_count
    success = result_set == expected_results
    print(success)
    print(result_set)
    print(expected_results)

    # Criando um novo DataFrame com os resultados
    new_data = {
        "Id": [name],
        "Resultado Obtido": [", ".join(sorted(result_set))],
        "Resultado Esperado": [", ".join(sorted(expected_results))],
        "Sucedido": [success],  # Usando 'FALSO' ou 'VERDADEIRO'
        "Momento Exato Da Primeira Inferencia Sucedida": [time_to_find],
        "OCR Options": [", ".join(map(str, sorted(options)))],
        "Tempo Processamento": [process_time],
        "Área dos Contornos": [", ".join(map(str, area_contours))],
        "Qntd Média de Contornos": [median_contour]
    }

    new_df = pd.DataFrame(new_data)
    save_or_update_table(new_df)

# TODO: RODAR ATE FIM DO VIDEO COM TODAS OPÇOES DE OCR A NOITE
for i in range(42, 137):
    try:
        run('Vídeo ' + str(i))
    except:
        continue
