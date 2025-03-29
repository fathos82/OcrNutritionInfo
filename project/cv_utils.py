import cv2
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
