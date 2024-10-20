# TODO: Pesquisar flake8, black, sphinx
from project.cv.preprocess_pass import PreProcessPass
from project.ocr.ocr_pass import OcrPass
from project.pipeline import Pipeline
from project.utils.image_data import ImageData

pipeline = Pipeline().add_passes(PreProcessPass, OcrPass)
pipeline.run(ImageData("/home/athos/testPython/OcrNutritionInfo/project/res/imagem-1.jpg"))
