import math
import os
import cv2
import numpy as np
import time
from structure.cv.find_contours import ContoursData
from structure.pipeline_factory import PipelineFactory, PipelineVariation
from structure.run_pipeline import run
from structure.runners.runner import Runner
from structure.utils.image_data import ImageData
from structure.utils.pandas_utils import contains_register, get_name_from_path, save_or_update_table


def hybrid_scoring_optimized(img,
                             min_dark_pixels=0.05,
                             min_laplacian=50,
                             max_light_pixels=0.9):
    """
    Versão otimizada que elimina resize e Laplacian duplicados.
    Retorna 0.0 para imagens descartáveis, senão retorna o score completo.
    """
    if img is None:
        return 0.0

    # --- Fase 1: Pré-processamento base (192x192) ---
    img = cv2.resize(img, (192, 192))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # --- Filtro Rápido (usando a imagem já redimensionada) ---
    # 1. Verifica excesso de pixels claros
    light_pixels = np.sum(gray > 200) / (gray.size + 1e-6)
    # if light_pixels > max_light_pixels:
    #     return 0.0, 0,0,0

    # 2. Verifica nitidez (Laplacian usado tanto para filtro quanto para score)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # if lap_var < min_laplacian:
    #     return 0.0, 0,0,0

    # 3. Verifica pixels escuros
    dark_pixels = np.sum(gray < 50) / (gray.size + 1e-6)
    # if dark_pixels < min_dark_pixels:
    #     return 0.0, 0,0,0

    # --- Fase 2: Score Completo (reaproveita variáveis já calculadas) ---
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    normalized = clahe.apply(gray)
    inverted = 255 - normalized
    blurred = cv2.GaussianBlur(inverted, (3, 3), 0)

    # Reaproveita o lap_var do filtro (evita recálculo)
    _, thresh = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark_ratio = np.sum(thresh == 0) / (thresh.size + 1e-6)
    edges = cv2.Canny(normalized, 30, 100)
    edge_density = np.sum(edges > 0) / (edges.size + 1e-6)

    score = (dark_ratio * 0.5) + (lap_var * 0.3) + (edge_density * 0.2)
    return float(score), dark_pixels, lap_var, light_pixels

def hybrid_scoring(img,
                  min_dark_pixels=0.05,
                  min_laplacian=50,
                  max_light_pixels=0.9,
                  fast_check_size=64):
    """
    Combina pré-filtro e cálculo de score em um único passo.
    Retorna 0.0 se a imagem for descartável, caso contrário, retorna o score completo.
    """
    if img is None:
        return 0.0

    # --- Fase 1: Pré-filtro Rápido (64x64) ---
    img_small = cv2.resize(img, (fast_check_size, fast_check_size))
    gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY) if len(img_small.shape) == 3 else img_small

    # 1. Descarta imagens com excesso de pixels claros (fundo branco)
    light_pixels = np.sum(gray > 200) / (gray.size + 1e-6)
    if light_pixels > max_light_pixels:
        return 0.0

    # 2. Descarta imagens borradas (Laplaciano)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < min_laplacian:
        return 0.0

    # 3. Descarta imagens sem pixels escuros (sem texto/fundo)
    dark_pixels = np.sum(gray < 50) / (gray.size + 1e-6)
    if dark_pixels < min_dark_pixels:
        return 0.0

    # --- Fase 2: Score Completo (192x192) ---
    # (Só executa se passar no pré-filtro)
    img = cv2.resize(img, (192, 192))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    normalized = clahe.apply(gray)
    inverted = 255 - normalized
    blurred = cv2.GaussianBlur(inverted, (3, 3), 0)
    lap_var = cv2.Laplacian(blurred, cv2.CV_64F).var()
    _, thresh = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark_ratio = np.sum(thresh == 0) / (thresh.size + 1e-6)
    edges = cv2.Canny(normalized, 30, 100)
    edge_density = np.sum(edges > 0) / (edges.size + 1e-6)

    score = (dark_ratio * 0.5) + (lap_var * 0.3) + (edge_density * 0.2)
    return float(score)

def prefilter_image(img, min_dark_pixels=0.05, min_laplacian=50, max_light_pixels=0.9):
    """
    Retorna False se a imagem for "lixo" (descarte imediato).
    """
    if img is None:
        return False

    # Redimensionamento rápido (64x64 para análise)
    img_small = cv2.resize(img, (64, 64))
    gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY) if len(img_small.shape) == 3 else img_small

    # 1. Verifica excesso de pixels claros (fundo branco)
    light_pixels = np.sum(gray > 200) / (gray.size + 1e-6)
    if light_pixels > max_light_pixels:
        return False

    # 2. Verifica falta de bordas (imagem borrada)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < min_laplacian:
        return False

    # 3. Verifica falta de pixels escuros (sem texto/fundo escuro)
    dark_pixels = np.sum(gray < 50) / (gray.size + 1e-6)
    if dark_pixels < min_dark_pixels:
        return False

    return True  # Passou no filtro

# todo: decidir porcentagem que ira ser vizualizada por ocr
#todo: melhor parametro de pieces
#todo: mexer no parametro de resize
# todo: testar pre-filter amanha
# todo: decidir se redimensiona


def optimized_deep_score(img):
    if img is None:
        return -1

    # 1. Redimensionamento menor (192x192 é um bom equilíbrio)
    img = cv2.resize(img, (192, 192))

    # 2. CLAHE mais rápido (menos tiles)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    normalized = clahe.apply(img)

    # 3. Blur reduzido (kernel 3x3)
    inverted = 255 - normalized
    blurred = cv2.GaussianBlur(inverted, (3, 3), 0)
    lap_var = cv2.Laplacian(blurred, cv2.CV_64F).var()

    # 4. Threshold OTSU (mantido)
    _, thresh = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark_ratio = np.sum(thresh == 0) / (thresh.size + 1e-6)

    # 5. Canny mais leve
    edges = cv2.Canny(normalized, 30, 100)
    edge_density = np.sum(edges > 0) / (edges.size + 1e-6)

    # Score (pesos mantidos)
    score = (dark_ratio * 0.5) + (lap_var * 0.3) + (edge_density * 0.2)
    return float(score)


def deep_score_image_for_text_detection(img):
    if img is None:
        return -1
    img = cv2.resize(img, (256, 256))
    # img = cv2.resize(img, (192, 192))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normalized = clahe.apply(img)
    inverted = 255 - normalized
    blurred = cv2.GaussianBlur(inverted, (5, 5), 0)
    lap_var = cv2.Laplacian(blurred, cv2.CV_64F).var()
    _, thresh = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark_ratio = np.sum(thresh == 0) / (thresh.size + 1e-6)
    edges = cv2.Canny(normalized, 50, 150)
    edge_density = np.sum(edges > 0) / (edges.size + 1e-6)
    score = (dark_ratio * 0.5) + (lap_var * 0.3) + (edge_density * 0.2)
    return float(score)

def score_image_for_text_detection(img):
    if img is None:
        return -1
    img = cv2.resize(img, (128, 128))
    inverted = 255 - img
    lap = cv2.Laplacian(inverted, cv2.CV_64F).var()
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    dark_pixels = np.sum(hist[:50])
    score = (dark_pixels * 0.6) + (lap * 0.4)
    return float(score)

class LocalRunner(Runner):
    def __init__(self, config):
        super().__init__(config)
        self.base_path = 'res/videos'
        # self.contours_crops = []


    def process(self,cap, start, end, pieces):
        max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("max frames:", max_frames)
        pieces_percent = (max_frames / pieces)
        start_frame = int(start * pieces_percent)
        end_frame = int(end * pieces_percent)
        print("start frame:", start_frame)
        print("end frame:", end_frame)


        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        contours_crops = []
        start_time = time.time()
        while cap.get(cv2.CAP_PROP_POS_FRAMES) != end_frame:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            # frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
            image_data = ImageData.from_image(frame)
            processed_data = self.pipeline.run(image_data)
            contours = processed_data.contours
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                crop = frame[y:y + h, x:x + w]
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

                # if prefilter_image(crop):  # <--- AQUI É O FILTRO!
                #     contours_crops.append(crop)
                contours_crops.append(crop)

        contours_sorted = [(crop, hybrid_scoring_optimized(crop)) for crop in contours_crops]
        contours_sorted.sort(key=lambda x: x[1][0], reverse=True)
        end_time = time.time()
        print("time: ", end_time - start_time)
        print(len(contours_sorted))
        for cnt, r in contours_sorted:
            cv2.imshow('crop', cnt)
            cv2.waitKey(0)
            print("imagem: ")
            print("dark_pixels: "+str(r[1]))
            print("lap_var: "+str(r[2]))
            print("light_pixels: "+str(r[3]))
            print()

        return False


    def run_pipeline(self, cap, pieces=2):
        start_time = time.time()
        end = math.ceil(pieces / 2)
        start = end - 1
        success = self.process(cap,start, end, pieces)
        previous_start = start
        raise RuntimeError()

        next_end = end
        if not success:
            while not success:
                if previous_start > 0:
                    previous_start = previous_start - 1
                    previous_end = previous_start + 1
                    success = self.process(cap,previous_start , previous_end, pieces)
                elif not success and next_end < pieces:
                    next_end = next_end + 1
                    next_start = next_end - 1
                    success = self.process(cap, next_start, next_end, pieces)
                else:
                    break
        print("--- %s seconds ---" % (time.time() - start_time))




    def load_videos_path(self):
        video_paths = []
        paths = os.listdir(self.base_path)
        for video_file in paths:
            if not video_file.endswith('.mp4') or contains_register(get_name_from_path(video_file)):
                continue
            video_path = os.path.join(self.base_path, video_file)
            video_paths.append(video_path)
        return video_paths

    def run(self):
        video_paths = self.load_videos_path()
        print(video_paths[9])
        cap = cv2.VideoCapture(video_paths[16])  # Assume o mesmo vídeo de teste
        self.run_pipeline(cap, pieces=5)
        #
        # # Coleta todos os recortes
        # start_time = time.time()
        # while cap.isOpened():
        #     ret, frame = cap.read()
        #     if not ret or frame is None:
        #         break
        #     image_data = ImageData.from_image(frame)
        #     processed_data = self.pipeline.run(image_data)
        #     contours = processed_data.contours
        #     for contour in contours:
        #         x, y, w, h = cv2.boundingRect(contour)
        #         crop = frame[y:y + h, x:x + w]
        #         crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        #         self.contours_crops.append(crop)
        # cap.release()
        # print('Total time:', time.time() - start_time)
        #
        #
        # # Medição de tempo + ordenação (Algoritmo Original)
        # start_time = time.time()
        # scored_original = [(crop, score_image_for_text_detection(crop)) for crop in self.contours_crops]
        # scored_original.sort(key=lambda x: x[1], reverse=True)
        # original_time = time.time() - start_time
        #
        # # Medição de tempo + ordenação (Algoritmo Atualizado)
        # start_time = time.time()
        # scored_deep = [(crop, deep_score_image_for_text_detection(crop)) for crop in self.contours_crops]
        # scored_deep.sort(key=lambda x: x[1], reverse=True)
        # deep_time = time.time() - start_time
        # # Medição de tempo + ordenação (Algoritmo Atualizado)
        # start_time = time.time()
        # scored_optimized_deep = [(crop, optimized_deep_score(crop)) for crop in self.contours_crops]
        # scored_optimized_deep.sort(key=lambda x: x[1], reverse=True)
        # scored_optimized_deep_time = time.time() - start_time
        #
        # # Exibe resultados
        # print(f"\n🔹 Algoritmo Original: {len(scored_original)} recortes | Tempo: {original_time:.4f}s")
        # print(f"🔹 Algoritmo Atualizado: {len(scored_deep)} recortes | Tempo: {deep_time:.4f}s")
        # print(f"🔹 Algoritmo optimized Atualizado: {len(scored_optimized_deep)} recortes | Tempo: {scored_optimized_deep_time:.4f}s")
        #
        # print(f"🔎 Diferença: {deep_time - original_time:.4f}s (+{(deep_time/original_time - 1)*100:.1f}%)")
        #
        # # Exemplo: Top 3 scores de cada algoritmo
        # print("\n🏆 Top 3 (Original):")
        # for i, (crop, score) in enumerate(scored_original[:3]):
        #     print(f"  {i+1}. Score: {score:.2f}")
        #
        # print("\n🏆 Top 3 (Atualizado):")
        # for i, (crop, score) in enumerate(scored_deep[:3]):
        #     print(f"  {i+1}. Score: {score:.2f}")
        #
        #
        # print("\n🏆 Top 3 (optimized):")
        # for i, (crop, score) in enumerate(scored_optimized_deep[:3]):
        #     print(f"  {i+1}. Score: {score:.2f}")



# 🔹 Algoritmo Original: 2592 recortes | Tempo: 0.4657s
# 🔹 Algoritmo Atualizado: 2592 recortes | Tempo: 2.0072s
# 🔎 Diferença: 1.5415s (+331.0%)
#
# 🏆 Top 3 (Original):
#   1. Score: 7282.38
#   2. Score: 7272.60
#   3. Score: 7133.19
#
# 🏆 Top 3 (Atualizado):
#   1. Score: 126.92
#   2. Score: 114.08
#   3. Score: 113.68



