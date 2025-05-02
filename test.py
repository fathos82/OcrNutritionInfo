import cv2
import numpy as np

cap = cv2.VideoCapture('res/videos/Vídeo 1.mp4')
while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normalized = clahe.apply(gray)
    inverted = 255 - normalized
    blurred = cv2.GaussianBlur(inverted, (5, 5), 0)
    lap_var = cv2.Laplacian(blurred, cv2.CV_64F).var()
    _, thresh = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark_ratio = np.sum(thresh == 0) / (thresh.size + 1e-6)
    edges = cv2.Canny(normalized, 50, 150)
    edge_density = np.sum(edges > 0) / (edges.size + 1e-6)

    cv2.imshow('Original', frame)
    cv2.imshow('CLAHE', normalized)
    cv2.imshow('Thresh', thresh)
    cv2.imshow('Edges', edges)
    cv2.waitKey(1)
cv2.destroyAllWindows()

