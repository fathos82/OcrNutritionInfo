import os

import cv2

from structure.utils.pandas_utils import get_name_from_path


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

def save_yolo_data_and_frame(crop_data, video_path, output_dir):
    """
    Salva a imagem do frame e o arquivo .txt no formato YOLO com as posições normalizadas do contorno.
    """
    cap = cv2.VideoCapture(video_path)
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    frame_id = int(crop_data[2])  # posição do frame
    contour = crop_data[1]

    x, y, w, h = cv2.boundingRect(contour)
    x_center = (x + w / 2) / frame_width
    y_center = (y + h / 2) / frame_height
    w_norm = w / frame_width
    h_norm = h / frame_height

    # Nome base (sem extensão) para salvar os arquivos
    base_name = f"{get_name_from_path(video_path)}_{frame_id}"

    # Salvar imagem completa do frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()
    if ret and frame is not None:
        img_path = os.path.join(output_dir, f"{base_name}.jpg")
        if not  os.path.exists(img_path):
            cv2.imwrite(img_path, frame)

        # Salvar anotação YOLO
        with open(os.path.join(output_dir, f"{base_name}.txt"), "a") as f:
            # Aqui usamos "0" como a classe, ajuste conforme necessário
            f.write(f"0 {x_center} {y_center} {w_norm} {h_norm}\n")

    cap.release()
