# TODO: Pesquisar flake8, black, sphinx
import cv2

from project.cv.filter_by_area import FilterByArea
from project.cv.filter_not_squares import FilterNotSquares
from project.cv.find_contours import FindContours, ContoursData
from project.cv.preprocess_pass import PreProcessPass
from project.ocr.ocr_pass import OcrPass
from project.pipeline import Pipeline, DetectionPass
from project.utils.image_data import ImageData

pipeline = Pipeline().add_passes(PreProcessPass, FindContours, FilterNotSquares)
data:ContoursData = pipeline.run(ImageData("project/res/1.jpg"))

cv2.imshow( "result", data.temporary_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
























































