"""
Módulo de Procesamiento en Frecuencia para Detección de Orientación y Deskewing (Nivel 5).
Permite:
1. Aplicar modulación de rayas periódicas horizontales.
2. Enderezar (deskew) las piezas rotadas antes del ensamblado.
"""

from typing import Tuple, Optional
import numpy as np
import cv2

__all__ = [
    "aplicar_filtro_rayas_horizontales",
    "estimar_orientacion_fourier",
    "estimar_orientacion_sobel",
    "enderezar_pieza",
    "rotar_imagen_ortogonal",
]

_MSG_EJERCICIO = "Hola, chismosin (again), fijate el contrato."

def aplicar_filtro_rayas_horizontales(
    imagen: np.ndarray,
    periodo: int = 8,
    amplitud: float = 0.38,
) -> np.ndarray:
    """
    Aplica una modulación armónica periódica horizontal con pico espectral dominante.
    Garantiza que en el dominio de Fourier 2D el componente de las rayas sea el
    pico más brillante e intenso globalmente fuera de la componente continua (DC).
    """
    alto, ancho = imagen.shape[:2]
    y_coords = np.arange(alto, dtype=np.float32)[:, None]

    patron_1d = (amplitud * np.cos(2.0 * np.pi * y_coords / float(periodo))).astype(np.float32)
    patron_2d = np.repeat(patron_1d, ancho, axis=1)

    if imagen.ndim == 3:
        patron_2d = patron_2d[:, :, None]

    es_float = issubclass(imagen.dtype.type, np.floating)
    img_f = imagen if es_float else imagen.astype(np.float32) / 255.0
    modulada = np.clip(img_f + patron_2d, 0.0, 1.0)
    if not es_float:
        modulada = (modulada * 255.0).astype(imagen.dtype)
    return modulada


def estimar_orientacion_fourier(
    imagen: np.ndarray,
    periodo_esperado: int = 8,
    radio_exclusion_dc: int = 15,
    mascara: Optional[np.ndarray] = None,
    radio_dc: Optional[int] = None,
) -> float:
    """
    Contrato esperado:
        Entrada: pieza rotada, con la modulación de rayas de
                 `aplicar_filtro_rayas_horizontales` y período `periodo_esperado`.
        Salida:  ángulo en grados que hay que pasarle a `enderezar_pieza` para
                 llevar la pieza a 0°.
    """
    raise NotImplementedError(_MSG_EJERCICIO)


def estimar_orientacion_sobel(
    imagen: np.ndarray,
    mascara: Optional[np.ndarray] = None,
    num_bins: int = 180,
) -> float:
    """
     Esto  no deberia funcar, si queres intentalo... padawan(?
    """
    raise NotImplementedError(_MSG_EJERCICIO)


def enderezar_pieza(
    imagen: np.ndarray,
    angulo_grados: float,
    padding: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rota la pieza para enderezarla alrededor de su centro geométrico.
    Si padding > 0, expande el lienzo con ese margen sobre fondo negro puro (0, 0, 0).

    Returns:
        (pieza_enderezada, mascara_enderezada)
    """
    alto, ancho = imagen.shape[:2]
    pad_h = padding
    pad_w = padding

    if pad_h > 0 or pad_w > 0:
        if imagen.ndim == 3:
            padded = np.pad(imagen, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode="constant", constant_values=0)
        else:
            padded = np.pad(imagen, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant", constant_values=0)
    else:
        padded = imagen.copy()

    ph, pw = padded.shape[:2]
    if abs(angulo_grados) < 0.2:
        rectificada = padded
    else:
        centro = (pw / 2.0, ph / 2.0)
        matriz_rot = cv2.getRotationMatrix2D(centro, -angulo_grados, 1.0)
        rectificada = cv2.warpAffine(
            padded,
            matriz_rot,
            (pw, ph),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    if issubclass(rectificada.dtype.type, np.floating):
        mask = (np.max(rectificada, axis=2) > 0.01).astype(np.uint8) * 255 if rectificada.ndim == 3 else (rectificada > 0.01).astype(np.uint8) * 255
    else:
        mask = (np.max(rectificada, axis=2) > 5).astype(np.uint8) * 255 if rectificada.ndim == 3 else (rectificada > 5).astype(np.uint8) * 255

    return rectificada, mask


def rotar_imagen_ortogonal(imagen: np.ndarray, grados_horarios: int) -> np.ndarray:
    """Rota una imagen 0°, 90°, 180° o 270° en sentido horario."""
    k = (grados_horarios % 360) // 90
    if k == 0:
        return imagen.copy()
    elif k == 1:
        return cv2.rotate(imagen, cv2.ROTATE_90_CLOCKWISE)
    elif k == 2:
        return cv2.rotate(imagen, cv2.ROTATE_180)
    elif k == 3:
        return cv2.rotate(imagen, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return imagen.copy()
