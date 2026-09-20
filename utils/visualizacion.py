"""
Módulo de Visualización para Jupyter Notebook y Google Colab.
Permite graficar de manera limpia y estética:
- Muestras de piezas desordenadas.
- Espectros 2D de Fourier y picos en frecuencia.
- Comparaciones lado a lado antes y después del filtrado.
- Grilla reconstruida vs. Imagen original.
"""

from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
import cv2

__all__ = [
    "mostrar_piezas_desordenadas",
    "mostrar_comparacion_imagen",
    "mostrar_espectro_fourier",
    "mostrar_reconstruccion",
    "mostrar_matriz_afinidad",
]


def mostrar_piezas_desordenadas(
    rompecabezas,
    max_piezas: int = 16,
    columnas_plot: int = 4,
    titulo: str = "Piezas Desordenadas (Entrada)",
) -> None:
    """Grafica una cuadrícula con las piezas desordenadas del rompecabezas."""
    piezas = rompecabezas.piezas[:max_piezas]
    n = len(piezas)
    filas_plot = int(np.ceil(n / columnas_plot))

    fig, axes = plt.subplots(filas_plot, columnas_plot, figsize=(3 * columnas_plot, 3 * filas_plot))
    axes = np.array(axes).reshape(-1)

    for i in range(len(axes)):
        if i < n:
            p = piezas[i]
            axes[i].imshow(np.clip(p, 0.0, 1.0))
            axes[i].set_title(f"Pieza #{i}", fontsize=10)
        axes[i].axis("off")

    plt.suptitle(f"{titulo} (Nivel {rompecabezas.metadatos.get("nivel","-")})", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


def mostrar_comparacion_imagen(
    original: np.ndarray,
    procesada: np.ndarray,
    titulo_orig: str = "Antes (Degradada)",
    titulo_proc: str = "Después (Filtrada / Procesada)",
    titulo_general: str = "Comparación",
) -> None:
    """Muestra dos imágenes lado a lado para evaluar el efecto de un filtro o procesamiento."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.imshow(np.clip(original, 0.0, 1.0))
    ax1.set_title(titulo_orig, fontsize=12)
    ax1.axis("off")

    ax2.imshow(np.clip(procesada, 0.0, 1.0))
    ax2.set_title(titulo_proc, fontsize=12)
    ax2.axis("off")

    plt.suptitle(titulo_general, fontsize=14)
    plt.tight_layout()
    plt.show()


def mostrar_espectro_fourier(
    imagen: np.ndarray,
    titulo: str = "Espectro 2D de Magnitud (Fourier)",
    cmap: str = "inferno",
) -> None:
    """Calcula y muestra el espectro de magnitud centrado 2D FFT en escala logarítmica."""
    if imagen.ndim == 3:
        gray = cv2.cvtColor((np.clip(imagen, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    else:
        gray = imagen.astype(np.float32)

    # Ventana de Hann para reducir fugas en bordes
    alto, ancho = gray.shape
    ventana = np.hanning(alto)[:, None] * np.hanning(ancho)[None, :]
    ventaneada = (gray - np.mean(gray)) * ventana

    f = np.fft.fft2(ventaneada)
    fshift = np.fft.fftshift(f)
    magnitud_log = np.log1p(np.abs(fshift))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.imshow(np.clip(imagen, 0.0, 1.0))
    ax1.set_title("Espacio Espacial", fontsize=12)
    ax1.axis("off")

    im = ax2.imshow(magnitud_log, cmap=cmap)
    ax2.set_title("Espacio de Frecuencia (log |F(u, v)|)", fontsize=12)
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    ax2.axis("off")

    plt.suptitle(titulo, fontsize=14)
    plt.tight_layout()
    plt.show()


def mostrar_reconstruccion(
    rompecabezas,
    grilla_propuesta: Optional[np.ndarray] = None,
    titulo: str = "Resultado del Reconstructor",
    piezas: Optional[List[np.ndarray]] = None,
) -> None:
    """Muestra la reconstrucción obtenida frente a la imagen base ground truth."""
    img_reconstruida = rompecabezas.pegar_piezas(grilla_propuesta, piezas=piezas)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    if rompecabezas.imagen_base is not None:
        ax1.imshow(np.clip(rompecabezas.imagen_base, 0.0, 1.0))
        ax1.set_title("Ground Truth (Imagen Original)", fontsize=12)
        ax1.axis("off")

    ax2.imshow(np.clip(img_reconstruida, 0.0, 1.0))
    ax2.set_title(f"{titulo} (Grilla {rompecabezas.cantidad_filas}x{rompecabezas.cantidad_columnas})", fontsize=12)
    ax2.axis("off")

    plt.tight_layout()
    plt.show()


def mostrar_matriz_afinidad(matriz: np.ndarray, titulo: str = "Matriz de Costo / Afinidad") -> None:
    """Muestra el mapa de calor de una matriz de costo (diagonal excluida/máscara)."""
    copia = matriz.copy()
    copia[np.isinf(copia)] = np.nan
    max_val = np.nanmax(copia)
    copia[np.isnan(copia)] = max_val * 1.1

    plt.figure(figsize=(6, 5))
    plt.imshow(copia, cmap="viridis")
    plt.colorbar(label="Costo (menor = mejor)")
    plt.title(titulo, fontsize=12)
    plt.xlabel("Pieza Propuesta (B)")
    plt.ylabel("Pieza Origen (A)")
    plt.tight_layout()
    plt.show()
