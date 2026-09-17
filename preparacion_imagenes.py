"""
Módulo de Preparación y Manejo de Imágenes Base para Rompecabezas.
Convención fundamental: Imágenes representadas en RGB, float64 en el rango [0.0, 1.0].
"""

import os
from pathlib import Path
from typing import Tuple, Union
import numpy as np
import cv2
from PIL import Image

__all__ = [
    "cargar_imagen",
    "guardar_imagen",
    "preparar_imagen_base",
    "asegurar_rgb_float",
    "float_a_uint8",
]


def asegurar_rgb_float(imagen: np.ndarray) -> np.ndarray:
    """
    Garantiza que la imagen sea un array de NumPy float64 en el rango [0.0, 1.0] con 3 canales RGB.
    """
    arr = np.asarray(imagen)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        # RGBA a RGB ignorando canal alfa si es opaco o componiendo sobre blanco/negro
        arr = arr[:, :, :3]

    if issubclass(arr.dtype.type, np.integer):
        max_val = float(np.iinfo(arr.dtype).max)
        arr = arr.astype(np.float64) / max_val
    elif issubclass(arr.dtype.type, np.floating):
        arr = arr.astype(np.float64)
        if arr.max() > 1.0 + 1e-4:
            arr = arr / 255.0

    return np.clip(arr, 0.0, 1.0)


def float_a_uint8(imagen: np.ndarray) -> np.ndarray:
    """Convierte una imagen float64 [0.0, 1.0] a uint8 [0, 255]."""
    clipeada = np.clip(imagen, 0.0, 1.0)
    return np.round(clipeada * 255.0).astype(np.uint8)


def cargar_imagen(ruta: Union[str, Path]) -> np.ndarray:
    """
    Carga una imagen desde el disco en formato RGB float64 [0.0, 1.0].
    """
    ruta_str = str(ruta)
    if not os.path.isfile(ruta_str):
        raise FileNotFoundError(f"No se encontró el archivo de imagen: {ruta_str}")

    img_bgr = cv2.imread(ruta_str, cv2.IMREAD_COLOR)
    if img_bgr is None:
        # Fallback con PIL
        pil_img = Image.open(ruta_str).convert("RGB")
        return asegurar_rgb_float(np.array(pil_img))

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return asegurar_rgb_float(img_rgb)


def guardar_imagen(ruta: Union[str, Path], imagen: np.ndarray) -> None:
    """
    Guarda una imagen en disco asegurando conversión limpia a PNG 8-bit.
    """
    ruta_path = Path(ruta)
    ruta_path.parent.mkdir(parents=True, exist_ok=True)

    uint8_rgb = float_a_uint8(imagen)
    uint8_bgr = cv2.cvtColor(uint8_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(ruta_path), uint8_bgr)


def preparar_imagen_base(
    ruta_origen: Union[str, Path],
    tamaño_objetivo: Tuple[int, int] = (1024, 1024),
    ruta_destino: Union[str, Path, None] = None,
) -> np.ndarray:
    """
    Recorta el centro de una imagen grande y la reduce con antialiasing (área / Lanczos)
    a una resolución base cuadrada (ej. 1024x1024).
    """
    img = cargar_imagen(ruta_origen)
    alto, ancho = img.shape[:2]
    alto_obj, ancho_obj = tamaño_objetivo

    lado_cuadrado = min(alto, ancho)
    inicio_y = (alto - lado_cuadrado) // 2
    inicio_x = (ancho - lado_cuadrado) // 2

    recorte_central = img[inicio_y : inicio_y + lado_cuadrado, inicio_x : inicio_x + lado_cuadrado]

    # Reducir con interpolación de área / bicúbica de alta calidad
    img_uint8 = float_a_uint8(recorte_central)
    img_redimensionada = cv2.resize(img_uint8, (ancho_obj, alto_obj), interpolation=cv2.INTER_AREA)
    resultado = asegurar_rgb_float(img_redimensionada)

    if ruta_destino is not None:
        guardar_imagen(ruta_destino, resultado)

    return resultado

def extraer_parche_cuadrado(matriz_imagen, longitud_lado_px, esquina_x=None, esquina_y=None):
    """
    Recorta una región cuadrada de tamaño 'longitud_lado_px x longitud_lado_px' píxeles de una imagen
    El origen del cuadrado será (esquina_x, esquina_y). Si son None, se centra la imagen
    """
    alto_imagen_px, ancho_imagen_px = matriz_imagen.shape[:2]
    lado_maximo_permitido_px = min(alto_imagen_px, ancho_imagen_px)

    #Validamos si se peude realizar el recorde
    if longitud_lado_px > lado_maximo_permitido_px:
        raise ValueError(
            f"El lado pedido ({longitud_lado_px}px) supera el tamaño máximo posible. "
            f"El recuadro máximo es de {lado_maximo_permitido_px}x{lado_maximo_permitido_px}px."
        )

    #Centramos el recorte en caso de que esquina_x o esquina_y sean None
    columna_inicio = (ancho_imagen_px - longitud_lado_px) // 2 if esquina_x is None else esquina_x
    fila_inicio = (alto_imagen_px - longitud_lado_px) // 2 if esquina_y is None else esquina_y

    #Calculamos el punto final del cuadrado
    columna_limite = columna_inicio + longitud_lado_px
    fila_limite = fila_inicio + longitud_lado_px

    #Validamos que el cuadrado no se va de lso bordes
    fuera_de_bordes = (
        columna_inicio < 0 or columna_limite > ancho_imagen_px or
        fila_inicio < 0 or fila_limite > alto_imagen_px
    )

    if fuera_de_bordes:
        raise ValueError(
            f"El recuadro de {longitud_lado_px}x{longitud_lado_px}px en posición "
            f"({columna_inicio}, {fila_inicio}) se sale de los límites de la imagen "
            f"({ancho_imagen_px}x{alto_imagen_px}px)."
        )

    #Extraemos parche y devolvemos copia
    return matriz_imagen[fila_inicio:fila_limite, columna_inicio:columna_limite].copy()