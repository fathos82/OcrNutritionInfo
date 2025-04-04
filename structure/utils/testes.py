# TODO: Pesquisar flake8, black, sphinx
from time import time

import cv2

from structure.cv.filter_by_area import FilterByArea
from structure.cv.filter_not_squares import FilterNotSquares
from structure.cv.find_contours import FindContours
from structure.cv.preprocess_pass import PreProcessPass
from structure.ocr.ocr_pass import OcrPass, OcrData
from structure.ocr.ocr_pass_tester import OcrPassTester
from structure.pipeline import Pipeline, DetectionPass
from structure.utils.image_data import ImageData

# TODO: Dividir em duas stretegias, rapida(FS) e devagar(FA)
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
            "ocr_options": [5],
            "r": "AÇUCAR ADICIONADO"
        },
        {
            "name_file": "2.mp4",
            "ocr_options": [5],
            "r": "SODIO"
        },
        {
            "name_file": "4.mp4",
            "ocr_options": [6,12],
            "r": ""

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

    finally:
        print("ALTO EM: " + str(word_set))
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def run_from_image(id):
    image_config_set = [
        {
            "name_file": "1.jpg",
            "ocr_options": [5,6,12]
        },
        {
            "name_file": "2.png",
            "ocr_options": [6, 12]
        },
        {
            "name_file": "3.jpeg",
            "ocr_options": [6, 12]
        }
    ]
    image_config = image_config_set[id]
    pipeline = Pipeline().add_passes(PreProcessPass, FindContours, FilterNotSquares, OcrPass(ocr_options=image_config["ocr_options"]))
    image = cv2.imread(f"project/res/images/{image_config['name_file']}")
    # image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
    data:OcrData = pipeline.run(ImageData.from_image(image))

    print("ALTO EM: "+str(data.word_set))
    cv2.waitKey(0)
