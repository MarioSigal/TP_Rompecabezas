"""
Módulo de Extracción y Matching Morfológico de Contornos y Encastres (Nivel 4).
Permite:
1. Binarizar piezas sobre fondo negro puro (0, 0, 0).
2. Segmentar el contorno en sus 4 lados (NORTE, ESTE, SUR, OESTE).
3. Clasificar cada borde en PLANO, SALIENTE (pestaña) o ENTRANTE (muesca/hendidura).
4. Medir la correlación de forma y compatibilidad de encastres.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import cv2

__all__ = [
    "binarize_piece",
    "extract_external_contour",
    "detect_jigsaw_corners",
    "detect_corners_and_split_sides",
    "segmentar_lados_pieza",
    "segmentar_borde_en_4",
    "pasar_borde_a_1d",
    "extraer_perfil_1d",
    "analyze_piece_shape",
    "compute_edge_correlation",
    "compute_jigsaw_shape_compatibility",
    "compatibilidad_forma",
    "calcular_mse_color_bordes",
]

_MSG_EJERCICIO = "Hola chismosin, fijate el contrato de la funcion guinio"


def binarize_piece(img: np.ndarray, fixed_threshold: float = 0.01) -> np.ndarray:
    """
    Binariza la imagen de la pieza aislando la silueta completa (255) del fondo negro (0).
    Aplica relleno morfológico de huecos interiores.
    Soporta float64 [0, 1] y uint8 [0, 255].
    """
    if issubclass(img.dtype.type, np.floating):
        thresh = fixed_threshold
    else:
        thresh = fixed_threshold * 255.0
 
    if img.ndim == 3:
        max_channel = np.max(img, axis=2)
    else:
        max_channel = img
 
    binary = (max_channel > thresh).astype(np.uint8) * 255

    h, w = binary.shape
    mask_flood = np.zeros((h + 2, w + 2), np.uint8)
    bin_inv = cv2.floodFill(binary.copy(), mask_flood, (0, 0), 255)[1]
    bin_filled = cv2.bitwise_not(bin_inv)
    binary = cv2.bitwise_or(binary, bin_filled)
 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
 
    return binary


def extract_external_contour(binary_mask: np.ndarray) -> np.ndarray:
    """Extrae el contorno exterior principal de mayor área."""
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No se detectó ningún contorno en la máscara binaria provista.")
    main_contour = max(contours, key=cv2.contourArea)
    return main_contour.squeeze(axis=1)


def detect_jigsaw_corners(contour_pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Detecta las 4 esquinas base (TL, TR, BR, BL) de la pieza."""
    mid_x, mid_y = np.mean(contour_pts, axis=0)
 
    q_tl = [p for p in contour_pts if p[0] <= mid_x and p[1] <= mid_y]
    q_tr = [p for p in contour_pts if p[0] >= mid_x and p[1] <= mid_y]
    q_br = [p for p in contour_pts if p[0] >= mid_x and p[1] >= mid_y]
    q_bl = [p for p in contour_pts if p[0] <= mid_x and p[1] >= mid_y]
 
    min_x, max_x = np.min(contour_pts[:, 0]), np.max(contour_pts[:, 0])
    min_y, max_y = np.min(contour_pts[:, 1]), np.max(contour_pts[:, 1])
 
    c_tl = min(q_tl, key=lambda p: (p[0] - min_x) ** 2 + (p[1] - min_y) ** 2)
    c_tr = min(q_tr, key=lambda p: (p[0] - max_x) ** 2 + (p[1] - min_y) ** 2)
    c_br = min(q_br, key=lambda p: (p[0] - max_x) ** 2 + (p[1] - max_y) ** 2)
    c_bl = min(q_bl, key=lambda p: (p[0] - min_x) ** 2 + (p[1] - max_y) ** 2)
 
    return np.array(c_tl), np.array(c_tr), np.array(c_br), np.array(c_bl)


def pasar_borde_a_1d(
    curva: np.ndarray,
    nombre_lado: str = "NORTE",
    num_muestras: int = 80,
) -> Dict[str, Any]:
    """
    Proyecta una curva 2D de contorno respecto a la recta que une sus extremos
    y extrae la señal 1D de desviación perpendicular.
    Clasifica el tipo de borde en: 'PLANO', 'SALIENTE' o 'ENTRANTE'.
    """
    p0 = curva[0].astype(np.float32)
    p1 = curva[-1].astype(np.float32)
    vec = p1 - p0
    length = float(np.linalg.norm(vec))
    if length == 0:
        return {
            "type": "PLANO",
            "profile": np.zeros(num_muestras, dtype=np.float32),
            "norm": 0.0,
            "length": 0.0,
            "max_dev": 0.0,
            "mean_dev": 0.0,
        }
 
    u = vec / length
    lado_u = nombre_lado.upper()
    if lado_u == "NORTE":
        normal_unit = np.array([0.0, -1.0])
    elif lado_u == "SUR":
        normal_unit = np.array([0.0, 1.0])
    elif lado_u == "OESTE":
        normal_unit = np.array([-1.0, 0.0])
    elif lado_u == "ESTE":
        normal_unit = np.array([1.0, 0.0])
    else:
        normal_unit = np.array([u[1], -u[0]], dtype=np.float32)
 
    rel = curva.astype(np.float32) - p0
    dev = np.dot(rel, normal_unit)
 
    t_orig = np.linspace(0, 1, len(curva))
    t_target = np.linspace(0, 1, num_muestras)
    profile = np.interp(t_target, t_orig, dev).astype(np.float32)
 
    max_dev = float(np.max(np.abs(profile)))
    mean_dev = float(np.mean(profile))
 
    if max_dev < length * 0.06:
        stype = "PLANO"
        profile = np.zeros(num_muestras, dtype=np.float32)
    elif mean_dev > 0:
        stype = "SALIENTE"
    else:
        stype = "ENTRANTE"
 
    norm = float(np.linalg.norm(profile))
    return {
        "type": stype,
        "profile": profile,
        "norm": norm,
        "length": length,
        "max_dev": max_dev,
        "mean_dev": mean_dev,
    }
 
 
extraer_perfil_1d = pasar_borde_a_1d


def _muestrear_color_curva(
    curva: np.ndarray,
    imagen_rgb: np.ndarray,
    binary_mask: Optional[np.ndarray] = None,
    num_samples: int = 80,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpola `num_samples` coordenadas a lo largo de `curva` y extrae el color RGB de `imagen_rgb`, 
    garantizando que no se tomen píxeles negros del fondo.
    Retorna (puntos_interpolados, color_profile).
    """
    t_orig = np.linspace(0, 1, len(curva))
    t_target = np.linspace(0, 1, num_samples)
    x_pts = np.interp(t_target, t_orig, curva[:, 0])
    y_pts = np.interp(t_target, t_orig, curva[:, 1])
 
    h, w = imagen_rgb.shape[:2]
    pts_interp = np.column_stack([x_pts, y_pts]).astype(np.float32)
 
    colores = []
    for x, y in zip(x_pts, y_pts):
        xi = int(np.clip(round(x), 0, w - 1))
        yi = int(np.clip(round(y), 0, h - 1))
 
        #Si el píxel cae justo en el fondo negro por redondeo, buscar en vecindad 3x3
        if binary_mask is not None and binary_mask[yi, xi] == 0:
            found = False
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = yi + dy, xi + dx
                    if 0 <= ny < h and 0 <= nx < w and binary_mask[ny, nx] > 0:
                        yi, xi = ny, nx
                        found = True
                        break
                if found:
                    break
 
        colores.append(imagen_rgb[yi, xi])
 
    color_arr = np.asarray(colores, dtype=np.float32)
    return pts_interp, color_arr
 
 
_SIDE_CACHE = {}


def detect_corners_and_split_sides(
    contour_pts: np.ndarray,
    binary_mask: Optional[np.ndarray] = None,
    num_samples: int = 80,
    imagen_rgb: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Segmenta el contorno en los 4 lados orientados (NORTE, ESTE, SUR, OESTE)
    y calcula la señal 1D de desviación perpendicular.
    """
    c_tl, c_tr, c_br, c_bl = detect_jigsaw_corners(contour_pts)
 
    def get_idx(pt):
        d = np.sum((contour_pts - pt) ** 2, axis=1)
        return int(np.argmin(d))
 
    i_tl = get_idx(c_tl)
    i_tr = get_idx(c_tr)
    i_br = get_idx(c_br)
    i_bl = get_idx(c_bl)
 
    n = len(contour_pts)
    shifted = np.roll(contour_pts, -i_tl, axis=0)
    i_tr_s = (i_tr - i_tl) % n
    i_br_s = (i_br - i_tl) % n
    i_bl_s = (i_bl - i_tl) % n
 
    if i_bl_s < i_tr_s:
        # Recorrido antihorario
        curve_w = shifted[0 : i_bl_s + 1]
        curve_s = shifted[i_bl_s : i_br_s + 1]
        curve_e = shifted[i_br_s : i_tr_s + 1][::-1]
        curve_n = np.vstack([shifted[i_tr_s:], shifted[0:1]])[::-1]
    else:
        # Recorrido horario
        curve_n = shifted[0 : i_tr_s + 1]
        curve_e = shifted[i_tr_s : i_br_s + 1]
        curve_s = shifted[i_br_s : i_bl_s + 1][::-1]
        curve_w = np.vstack([shifted[i_bl_s:], shifted[0:1]])[::-1]
 
    info_n = pasar_borde_a_1d(curve_n, "NORTE", num_muestras=num_samples)
    info_e = pasar_borde_a_1d(curve_e, "ESTE", num_muestras=num_samples)
    info_s = pasar_borde_a_1d(curve_s, "SUR", num_muestras=num_samples)
    info_w = pasar_borde_a_1d(curve_w, "OESTE", num_muestras=num_samples)
 
    #Perfil de color a lo largo de cada costura
    #Sigue la CURVA del encastre, no una columna recta: por eso hace falta la imagen ademas del contorno.
    if imagen_rgb is not None:
        for nombre_lado, curva, info in (("NORTE", curve_n, info_n), ("ESTE", curve_e, info_e),
                                         ("SUR", curve_s, info_s), ("OESTE", curve_w, info_w)):
            puntos, colores = _muestrear_color_curva(curva, imagen_rgb, binary_mask, num_samples)
            info["color_profile"] = colores
            info["curve"] = puntos
 
    types = [info_n["type"], info_e["type"], info_s["type"], info_w["type"]]
    num_flat = sum(1 for t in types if t == "PLANO")
    topology = "CORNER" if num_flat == 2 else ("BORDER" if num_flat == 1 else "INTERIOR")
    tipo_pieza = "ESQUINA" if num_flat == 2 else ("LADO" if num_flat == 1 else "INTERIOR")
 
    return {
        "NORTE": info_n,
        "ESTE": info_e,
        "SUR": info_s,
        "OESTE": info_w,
        "topology": topology,
        "tipo_pieza": tipo_pieza,
        "num_flat": num_flat,
        "num_planos": num_flat,
        "corners": {"TL": c_tl, "TR": c_tr, "BR": c_br, "BL": c_bl},
    }
 
 
#Alias en español para el trabajo práctico de los estudiantes
segmentar_lados_pieza = detect_corners_and_split_sides
segmentar_borde_en_4 = detect_corners_and_split_sides


def analyze_piece_shape(img: np.ndarray) -> Dict[str, Any]:
    """Analiza la silueta morfológica de una pieza."""
    binary = binarize_piece(img)
    contour = extract_external_contour(binary)
    sides_info = detect_corners_and_split_sides(contour, binary, imagen_rgb=img)
    return {
        "binary_mask": binary,
        "contour": contour,
        "sides": sides_info,
        "topology": sides_info["topology"],
    }


def compute_edge_correlation(side_a: Dict[str, Any], side_b: Dict[str, Any]) -> float:
    """
    Contrato esperado:
        Entrada: dos diccionarios de lado, tal como los devuelve `segmentar_borde_en_4`
        Salida:  float en [0.0, 1.0]. 1.0 = encastre complementario perfecto,
                 0.0 = incompatibles.
    """
    raise NotImplementedError(_MSG_EJERCICIO)


def compute_jigsaw_shape_compatibility(side_a: Dict[str, Any], side_b: Dict[str, Any]) -> float:
    """
     costo morfológico de encastre entre dos lados.
 
    Contrato esperado:
        Salida: float, donde 0.0 es el encastre óptimo y un valor muy grande
                (p. ej. 1e5) marca un par incompatible.
    """
    raise NotImplementedError(_MSG_EJERCICIO)


def calcular_mse_color_bordes(side_a: Dict[str, Any], side_b: Dict[str, Any]) -> float:
    """
    Calcula el Error Cuadrático Medio (MSE) entre los perfiles de color de dos bordes enfrentados.
    Los perfiles corren orientados en el mismo sentido a lo largo de la costura.
    """
    col_a = side_a.get("color_profile")
    col_b = side_b.get("color_profile")
    if col_a is None or col_b is None:
        return 0.0
 
    diff = col_a - col_b
    return float(np.mean(diff ** 2))


# Caché para no recalcular la forma de la misma pieza repetidamente
_SHAPE_CACHE = {}


def _get_cached_shape(pieza: np.ndarray, cache_id: Optional[int] = None) -> Dict[str, Any]:
    global _SHAPE_CACHE
    if cache_id is not None and cache_id in _SHAPE_CACHE:
        return _SHAPE_CACHE[cache_id]
    if len(_SHAPE_CACHE) > 2000:
        _SHAPE_CACHE.clear()
    shape_info = analyze_piece_shape(pieza)
    if cache_id is not None:
        _SHAPE_CACHE[cache_id] = shape_info
    return shape_info


def compatibilidad_forma(
    pieza_a: np.ndarray,
    pieza_b: np.ndarray,
    relacion: str = "horizontal",
) -> float:
    """
     compatibilidad de forma entre dos piezas enteras.
 
    Contrato esperado:
        relacion='horizontal': B va a la derecha de A.
        relacion='vertical':   B va abajo de A.
        Salida: float, menor = mejor encastre.
 
    Sugerencia: Si llageste hasta aca, te reomiendo `_get_cached_shape` para evitar recalcular la silueta de la misma pieza
    en cada comparación
    """
    raise NotImplementedError(_MSG_EJERCICIO)

