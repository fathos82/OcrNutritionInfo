import cv2
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


video_path = 'project/res/videos/Vídeo 8.mp4'

cap = cv2.VideoCapture(video_path)
skip_frames  = 2
count_frame = 0

pipeline = Pipeline().add_passes(Cropper(0.8), BlackFilter, FilterMaskByArea, FindContours, FilterByArea, ExtractPossibilities)


pipelines ={
    'BlackSearcher': [],

}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    count_frame += 1
    if count_frame % skip_frames == 0:
        continue
    cv2.waitKey(0)
    result:PossibilitiesData = pipeline.run(ImageData.from_image(frame))
    pipelines['BlackSearcher'].extend(result.possibilities)







