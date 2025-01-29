# TODO: Pesquisar flake8, black, sphinx
from time import time

import cv2
from project.ocr.ocr_pass_tester import OcrPassTester
from project.pipeline import Pipeline, DetectionPass
from project.utils.image_data import ImageData


#TODO: MultThreading Partion,
# Change Lib To Process Video,
# IMPROVE FILTERING,
# RESIZE IMAGEM, CROP
# IMAGE (CENTER)
# APLYING N TESSERECAT CONFIG
pipeline = Pipeline().add_passes(OcrPassTester) # TODO: CRIAR PASSO RESIZE
entry_data = ImageData("project/res/contour_15.png")
print(entry_data)
data = pipeline.run(entry_data)
print(data)
# cap = cv2.VideoCapture("res/contour_8.png")
# ret, frame = cap.read()
# frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
# data: OcrData = pipeline.run(ImageData.from_image(frame))
# word_set = set()
# start_total = time()
# try:
#     while cap.isOpened():
#         if not ret:
#             print("Can't read frame")
#             break
#         # if cv2.waitKey(1) & 0xFF == ord('r'):
#         start = time()
#         ret, frame = cap.read()
#         frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
#         end = time()
#         # print("Time to Read Image: ", end - start)
#         start = time()
#         data: OcrData = pipeline.run(ImageData.from_image(frame))
#         if len(word_set) > 0:
#             raise Exception()
#         end = time()
#         # print("Processing time: ", end - start)
#         word_set.update(data.word_set)
#         # for count in data.contours:
#         #     x, y, w, h = cv2.boundingRect(count)
#         #     cv2.rectangle(data.temporary_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
#         # cv2.imshow("result", frame)
# except Exception as e:
#     end_total = time()
#     print("Total time:", end_total - start_total)
#     print(word_set)
# finally:
#     print(word_set)
cv2.waitKey(0)
cv2.destroyAllWindows()
























































