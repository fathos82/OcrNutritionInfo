import cv2
from msgpack.fallback import BytesIO
from snowballstemmer import algorithms

from project.cv.black_filter import BlackFilter
from project.cv.cropper import Cropper
from project.cv.extract_contours import ExtractPossibilities, PossibilitiesData
from project.cv.filter_mask_by_area import FilterMaskByArea
from project.cv.find_contours import FindContours
from project.cv.filter_by_area import FilterByArea
from project.ocr.ocr_pass import OcrPass
from project.pipeline import Pipeline
from project.utils.image_data import ImageData


def resize_image(image, new_width=720):
    """Redimensiona a imagem mantendo a proporção."""
    height, width = image.shape[:2]
    new_height = int(height * new_width / width)
    return cv2.resize(image, (new_width, new_height))

def capture_frame(frame, time_code, file_path):
    file_name = file_path.split('/')[-1]
    file_name = file_name.split('.')[0]
    print(file_name)
    cv2.imwrite(f"project/res/frames/{file_name}_{time_code}.jpg", frame)

def get_time_code(time_ms):
    seconds = int(time_ms / 1000)
    minutes = int(seconds / 60)
    hours = int(minutes / 60)
    return f"{hours:02d}_{minutes%60:02d}_{seconds%60:02d}_{int(time_ms):02d}"





# Video 02 sucesso

video_path = 'project/res/videos/Vídeo 1.mp4'

cap = cv2.VideoCapture(video_path)
skip_frames  = 2
count_frame = 0

pipeline = Pipeline().add_passes(Cropper(0.8), BlackFilter, FilterMaskByArea, FindContours, FilterByArea, OcrPass)


# pipelines ={
#     'BlackSearcher': [],
#
# }

black_searcher = []

while cap.isOpened():
    tick = cv2.getTickCount()
    ret, frame = cap.read()
    if not ret:
        print('Video not found.')
        break
    result = pipeline.run(ImageData.from_image(resize_image(frame)))

    # count_frame += 1
    # if count_frame % skip_frames == 0:
    #     continue
    # cv2.imshow('Video', frame)
    # k = cv2.waitKey(0) & 0xFF
    # if k == ord('c'):
    #     print(cap.get(cv2.CAP_PROP_POS_MSEC))
    #
    #     capture_frame(frame, get_time_code(cap.get(cv2.CAP_PROP_POS_MSEC)), video_path)
    # if k == ord('q') :
    #     break

    result:PossibilitiesData = pipeline.run(ImageData.from_image(frame))
    black_searcher.extend(result.possibilities)

print(len(black_searcher))







# Dados para Teste
# Resultados do Dados
# Selecionar Frames
# Coletar Informações


# Nome do Video
# Media de Contornos
# Momento Inferencia Bem Sucedida
# Opção de OCR utilizada
