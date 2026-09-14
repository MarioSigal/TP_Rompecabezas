"""
Módulo de Procesamiento en Frecuencia para Detección de Orientación y Deskewing (Nivel 5).
Permite:
1. Aplicar modulación de rayas periódicas horizontales.
2. Estimar el ángulo de rotación mediante la Transformada 2D de Fourier (FFT) o gradientes Sobel.
3. Enderezar (deskew) las piezas rotadas antes del ensamblado.
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
    Estima el ángulo de inclinación mediante el pico espectral dominante en Fourier 2D.

    Principios físicos:
    1. Las rayas periódicas generan un pico armónico brillante simétrico respecto al centro DC.
    2. Al rotar la pieza un ángulo theta, el pico se desplaza sobre una circunferencia de radio R = alto / periodo.
    3. Anulando la componente continua (centro DC), el valor máximo de magnitud (np.argmax) identifica directamente la coordenada del pico.
    4. El ángulo de rotación se obtiene mediante la relación trigonométrica con arctan2.
    Retorna el ángulo en grados necesario para enderezar la pieza a 0°.
    """
    if radio_dc is not None:
        radio_exclusion_dc = radio_dc
    if imagen.ndim == 3:
        if issubclass(imagen.dtype.type, np.floating):
            gray = cv2.cvtColor((np.clip(imagen, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        else:
            gray = cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY).astype(np.float32)
    else:
        gray = imagen.astype(np.float32)

    # 1. Transformada 2D de Fourier centrada
    F = np.fft.fftshift(np.fft.fft2(gray))
    magnitud = np.abs(F)

    # 2. Anular la componente continua (DC)
    cy, cx = magnitud.shape[0] // 2, magnitud.shape[1] // 2
    r_dc = radio_exclusion_dc
    magnitud[cy - r_dc : cy + r_dc + 1, cx - r_dc : cx + r_dc + 1] = 0.0

    # 3. Encontrar el pico más intenso en la circunferencia
    yp, xp = np.unravel_index(np.argmax(magnitud), magnitud.shape)
    dy = yp - cy
    dx = xp - cx

    # Tomar semiplano superior por simetría conjugada de Fourier
    if dy > 0:
        dy, dx = -dy, -dx

    # 4. Calcular el ángulo para enderezar (contrarrestar la inclinación)
    angulo_correccion = float(-np.degrees(np.arctan2(dx, -dy)))
    return angulo_correccion


def estimar_orientacion_sobel(
    imagen: np.ndarray,
    mascara: Optional[np.ndarray] = None,
    num_bins: int = 180,
) -> float:
    """
    Estima el ángulo de inclinación mediante el histograma ponderado de direcciones
    de gradiente espacial Sobel.
    """
    if imagen.ndim == 3:
        if issubclass(imagen.dtype.type, np.floating):
            gray = cv2.cvtColor((imagen * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        else:
            gray = cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY).astype(np.float32)
    else:
        gray = imagen.astype(np.float32)

    if mascara is not None:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        eroded_mask = cv2.erode(mascara, kernel)
    else:
        eroded_mask = (gray > 5.0).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        eroded_mask = cv2.erode(eroded_mask, kernel)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    magnitud = np.sqrt(gx ** 2 + gy ** 2)
    angulo_deg = -np.rad2deg(np.arctan2(gy, gx))
    angulo_axial = np.mod(angulo_deg, 180.0)

    validos = (eroded_mask > 0) & (magnitud > 10.0)
    if not np.any(validos):
        return 0.0

    conteos, bordes_bin = np.histogram(
        angulo_axial[validos],
        bins=num_bins,
        range=(0.0, 180.0),
        weights=magnitud[validos],
    )

    pico = np.argmax(conteos)
    angulo_normal = 0.5 * (bordes_bin[pico] + bordes_bin[pico + 1])
    angulo_raya = angulo_normal - 90.0

    if angulo_raya > 90.0:
        angulo_raya -= 180.0
    elif angulo_raya < -90.0:
        angulo_raya += 180.0

    return float(angulo_raya)


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
