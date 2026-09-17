"""
Módulo de Métricas Oficiales de Evaluación del TP.
Evalúa:
1. Calidad del Filtrado: PSNR y SSIM frente a la imagen base limpia.
2. Calidad de Compatibilidad: Precisión Top-1 y MRR (Mean Reciprocal Rank) sobre las matrices de costo.
3. Calidad de Reconstrucción: Precisión de Vecindad (Neighbor Accuracy) y Precisión Directa (Direct Placement).
"""

from typing import Dict, Any, Optional
import numpy as np
try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    _TIENE_SKIMAGE = True
except ImportError:
    _TIENE_SKIMAGE = False

__all__ = [
    "calcular_psnr",
    "calcular_ssim",
    "calcular_precision_top1",
    "calcular_rango_reciproco_medio",
    "calcular_precision_directa",
    "calcular_precision_vecindad",
    "generar_reporte_completo",
    "imprimir_reporte",
]


# ==============================================================================
# Calidad del Filtrado (PSNR / SSIM)
# ==============================================================================

def calcular_psnr(imagen_referencia: np.ndarray, imagen_estimada: np.ndarray) -> float:
    """Calcula el PSNR (Peak Signal-to-Noise Ratio) en dB entre la imagen base y la filtrada."""
    ref = np.clip(np.asarray(imagen_referencia, dtype=np.float64), 0.0, 1.0)
    est = np.clip(np.asarray(imagen_estimada, dtype=np.float64), 0.0, 1.0)
    if _TIENE_SKIMAGE:
        return float(peak_signal_noise_ratio(ref, est, data_range=1.0))
    mse = np.mean((ref - est) ** 2)
    if mse <= 1e-12:
        return 100.0
    return float(10.0 * np.log10(1.0 / mse))


def calcular_ssim(imagen_referencia: np.ndarray, imagen_estimada: np.ndarray) -> float:
    """Calcula el Índice de Similitud Estructural (SSIM) en el rango [-1, 1]."""
    ref = np.clip(np.asarray(imagen_referencia, dtype=np.float64), 0.0, 1.0)
    est = np.clip(np.asarray(imagen_estimada, dtype=np.float64), 0.0, 1.0)
    if _TIENE_SKIMAGE:
        eje_canal = -1 if ref.ndim == 3 else None
        return float(structural_similarity(ref, est, data_range=1.0, channel_axis=eje_canal))
    # Fallback aproximado si no está skimage
    c1 = (0.01) ** 2
    c2 = (0.03) ** 2
    mu1 = np.mean(ref)
    mu2 = np.mean(est)
    var1 = np.var(ref)
    var2 = np.var(est)
    cov12 = np.mean((ref - mu1) * (est - mu2))
    ssim = ((2 * mu1 * mu2 + c1) * (2 * cov12 + c2)) / ((mu1**2 + mu2**2 + c1) * (var1 + var2 + c2))
    return float(np.clip(ssim, -1.0, 1.0))


# ==============================================================================
# Calidad de Compatibilidad (Top-1 / MRR)
# ==============================================================================

def _calcular_posiciones_en_ranking(matriz_afinidad: np.ndarray, adyacencias_reales: Dict[int, int]):
    posiciones = []
    for id_pieza, id_vecino_real in adyacencias_reales.items():
        if id_pieza >= matriz_afinidad.shape[0] or id_vecino_real >= matriz_afinidad.shape[1]:
            continue
        costos = matriz_afinidad[id_pieza]
        costo_del_verdadero = costos[id_vecino_real]
        cantidad_mejores = int(np.sum(costos < costo_del_verdadero))
        posiciones.append(cantidad_mejores + 1)
    return posiciones


def calcular_precision_top1(matrices_afinidad: Dict[str, np.ndarray], rompecabezas) -> Dict[str, float]:
    """
    Calcula el porcentaje de casos donde el vecino verdadero quedó en el 1er lugar del ranking (menor costo).
    """
    adyacencias = rompecabezas.obtener_adyacencias_correctas()
    resultado = {}

    for relacion in ("horizontal", "vertical"):
        posiciones = _calcular_posiciones_en_ranking(matrices_afinidad[relacion], adyacencias[relacion])
        resultado[relacion] = float(np.mean([pos == 1 for pos in posiciones])) if posiciones else 0.0

    resultado["promedio"] = float(np.mean([resultado["horizontal"], resultado["vertical"]]))
    return resultado


def calcular_rango_reciproco_medio(matrices_afinidad: Dict[str, np.ndarray], rompecabezas) -> float:
    """
    Calcula el Mean Reciprocal Rank (MRR) del vecino correcto (1.0 si es 1ro, 0.5 si es 2do, etc.).
    """
    adyacencias = rompecabezas.obtener_adyacencias_correctas()
    inversos = []

    for relacion in ("horizontal", "vertical"):
        posiciones = _calcular_posiciones_en_ranking(matrices_afinidad[relacion], adyacencias[relacion])
        inversos.extend(1.0 / pos for pos in posiciones)

    return float(np.mean(inversos)) if inversos else 0.0


# ==============================================================================
# Calidad de Reconstrucción (Vecindad / Directa)
# ==============================================================================

def calcular_precision_directa(grilla_propuesta: np.ndarray, rompecabezas) -> float:
    """
    Porcentaje de piezas posicionadas en la celda (fila, columna) exacta de la solución original.
    """
    grilla_correcta = np.asarray(rompecabezas.obtener_matriz_correcta())
    grilla_propuesta = np.asarray(grilla_propuesta)
    if grilla_correcta.shape != grilla_propuesta.shape:
        return 0.0
    return float(np.mean(grilla_propuesta == grilla_correcta))


def calcular_precision_vecindad(grilla_propuesta: np.ndarray, rompecabezas) -> float:
    """
    Porcentaje de pares de bordes contiguos (horizontal y vertical) que coinciden con los vecinos reales.
    Invariante a traslaciones globales en el ensamblado.
    """
    adyacencias = rompecabezas.obtener_adyacencias_correctas()
    grilla = np.asarray(grilla_propuesta)
    cantidad_filas, cantidad_columnas = grilla.shape

    aciertos = 0
    total = 0

    for fila in range(cantidad_filas):
        for columna in range(cantidad_columnas):
            id_pieza = int(grilla[fila, columna])
            if id_pieza < 0:
                continue

            # Vecino a la derecha
            if columna + 1 < cantidad_columnas:
                total += 1
                vecino_propuesto = int(grilla[fila, columna + 1])
                if adyacencias["horizontal"].get(id_pieza, -1) == vecino_propuesto:
                    aciertos += 1

            # Vecino abajo
            if fila + 1 < cantidad_filas:
                total += 1
                vecino_propuesto = int(grilla[fila + 1, columna])
                if adyacencias["vertical"].get(id_pieza, -1) == vecino_propuesto:
                    aciertos += 1

    return float(aciertos / total) if total else 0.0


# ==============================================================================
# Generación e Impresión de Reportes
# ==============================================================================

def generar_reporte_completo(
    rompecabezas,
    matrices_afinidad: Optional[Dict[str, np.ndarray]] = None,
    grilla_propuesta: Optional[np.ndarray] = None,
    imagen_limpiada: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Genera un reporte consolidado con todas las métricas calculadas."""
    reporte = {}

    if matrices_afinidad is not None:
        top1 = calcular_precision_top1(matrices_afinidad, rompecabezas)
        reporte["top1_horizontal"] = top1["horizontal"]
        reporte["top1_vertical"] = top1["vertical"]
        reporte["top1_promedio"] = top1["promedio"]
        reporte["mrr"] = calcular_rango_reciproco_medio(matrices_afinidad, rompecabezas)

    if grilla_propuesta is not None:
        reporte["precision_vecindad"] = calcular_precision_vecindad(grilla_propuesta, rompecabezas)
        reporte["precision_directa"] = calcular_precision_directa(grilla_propuesta, rompecabezas)

    if imagen_limpiada is not None and getattr(rompecabezas, "imagen_base", None) is not None:
        reporte["psnr"] = calcular_psnr(rompecabezas.imagen_base, imagen_limpiada)
        reporte["ssim"] = calcular_ssim(rompecabezas.imagen_base, imagen_limpiada)

    return reporte


def imprimir_reporte(reporte: Dict[str, Any], titulo: str = "") -> None:
    """Muestra el reporte de métricas en consola o notebook con formato legible."""
    if titulo:
        print(f"\n{'=' * 15} {titulo} {'=' * 15}")

    claves_porcentuales = (
        "top1_horizontal",
        "top1_vertical",
        "top1_promedio",
        "mrr",
        "precision_vecindad",
        "precision_directa",
    )

    for clave, valor in reporte.items():
        if clave in claves_porcentuales:
            print(f"  {clave:22s} {valor:7.1%}")
        elif clave == "psnr":
            print(f"  {clave:22s} {valor:7.2f} dB")
        elif clave == "ssim":
            print(f"  {clave:22s} {valor:7.4f}")
        else:
            print(f"  {clave:22s} {valor}")
