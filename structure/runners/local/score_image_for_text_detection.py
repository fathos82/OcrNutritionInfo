import cv2
import numpy as np


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
    if light_pixels > max_light_pixels:
        return 0.0, 0,0,0

    # 2. Verifica nitidez (Laplacian usado tanto para filtro quanto para score)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < min_laplacian:
        return 0.0, 0,0,0

    # 3. Verifica pixels escuros
    dark_pixels = np.sum(gray < 50) / (gray.size + 1e-6)
    if dark_pixels < min_dark_pixels:
        return 0.0, 0,0,0

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