"""
Versiones de las funciones de utils/visualizacion.py, utils/animacion.py y
core/metricas.py adaptadas a RompecabezasV2 (piezas como objetos Pieza con
forma real, sin imagen_base ni el parametro piezas= de pegar_piezas).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import matplotlib.pyplot as plt

from core.RompecabezasV2 import RompecabezasV2
from core.metricas import (
    calcular_precision_top1,
    calcular_rango_reciproco_medio,
    calcular_precision_vecindad,
    calcular_precision_directa,
    calcular_psnr,
    calcular_ssim,
)
from utils.animacion import (
    _a_bytes,
    _componer_cuadro,
    _dibujar_grilla_vacia,
    _escalar_por_bloques,
    _marcar_celda,
)

__all__ = [
    "mostrar_piezas_desordenadas_v2",
    "mostrar_reconstruccion_v2",
    "generar_reporte_completo_v2",
    "renderizar_pasos_v2",
    "crear_animacion_v2",
]


def mostrar_piezas_desordenadas_v2(
    rompecabezas: RompecabezasV2,
    max_piezas: int = 16,
    columnas_plot: int = 4,
    titulo: str = "Piezas Desordenadas (Entrada)",
) -> None:
    """Grafica una cuadrícula con las piezas (con su forma real, no rectangular) de un RompecabezasV2."""
    piezas = rompecabezas.piezas[:max_piezas]
    n = len(piezas)
    filas_plot = int(np.ceil(n / columnas_plot))

    fig, axes = plt.subplots(filas_plot, columnas_plot, figsize=(3 * columnas_plot, 3 * filas_plot))
    axes = np.array(axes).reshape(-1)

    for i in range(len(axes)):
        if i < n:
            pieza = piezas[i]
            axes[i].imshow(pieza.devolver_pieza())
            axes[i].set_title(f"Pieza #{pieza.id}", fontsize=10)
        axes[i].axis("off")

    plt.suptitle(f"{titulo} (Nivel {rompecabezas.metadatos.get('nivel', '-')})", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


def mostrar_reconstruccion_v2(
    rompecabezas: RompecabezasV2,
    grilla_propuesta: Optional[np.ndarray] = None,
    titulo: str = "Resultado del Reconstructor",
) -> None:
    """Muestra la reconstrucción obtenida frente a la imagen base ground truth (RompecabezasV2)."""
    img_reconstruida = rompecabezas.pegar_piezas(grilla_propuesta)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # RompecabezasV2 no tiene imagen_base: la imagen previa a cortar y degradar por pieza
    # queda guardada en metadatos["imagen"].
    imagen_base = rompecabezas.metadatos.get("imagen")
    if imagen_base is not None:
        ax1.imshow(np.clip(imagen_base, 0.0, 1.0))
        ax1.set_title("Ground Truth (Imagen Original)", fontsize=12)
        ax1.axis("off")

    ax2.imshow(np.clip(img_reconstruida, 0.0, 1.0))
    ax2.set_title(f"{titulo} (Grilla {rompecabezas.cantidad_filas}x{rompecabezas.cantidad_columnas})", fontsize=12)
    ax2.axis("off")

    plt.tight_layout()
    plt.show()


def generar_reporte_completo_v2(
    rompecabezas: RompecabezasV2,
    matrices_afinidad: Optional[Dict[str, np.ndarray]] = None,
    grilla_propuesta: Optional[np.ndarray] = None,
    imagen_limpiada: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Genera un reporte consolidado con todas las métricas calculadas (RompecabezasV2)."""
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

    imagen_base = rompecabezas.metadatos.get("imagen")
    if imagen_limpiada is not None and imagen_base is not None:
        reporte["psnr"] = calcular_psnr(imagen_base, imagen_limpiada)
        reporte["ssim"] = calcular_ssim(imagen_base, imagen_limpiada)

    return reporte


def renderizar_pasos_v2(
    rompecabezas: RompecabezasV2,
    historial: List[Dict[str, Any]],
    escala: int = 2,
    resaltar_ultima: bool = True,
) -> List[np.ndarray]:
    filas = rompecabezas.cantidad_filas
    cols = rompecabezas.cantidad_columnas

    # El tamaño del tablero ya lo conocemos por self.imagen, no hace falta pegar piezas para eso.
    alto_original, ancho_original = rompecabezas.imagen.shape[:2]
    alto_red = alto_original // max(escala, 1)
    ancho_red = ancho_original // max(escala, 1)

    fondo = _dibujar_grilla_vacia(alto_red, ancho_red, filas, cols)
    grilla_parcial = np.full((filas, cols), -1, dtype=int)
    cuadros = [fondo.copy()]

    for paso in historial:
        grilla_parcial[paso["fila"], paso["columna"]] = int(paso["id_pieza"])

        # Generar tablero parcial
        tablero = rompecabezas.pegar_piezas(grilla_parcial)
        tablero_red = _escalar_por_bloques(tablero, escala)

        cuadro = fondo.copy()
        # En las celdas ocupadas colocamos la imagen
        alto_c = alto_red / filas
        ancho_c = ancho_red / cols

        for r in range(filas):
            for c in range(cols):
                if grilla_parcial[r, c] != -1:
                    y0 = int(round(r * alto_c))
                    y1 = min(int(round((r + 1) * alto_c)), alto_red)
                    x0 = int(round(c * ancho_c))
                    x1 = min(int(round((c + 1) * ancho_c)), ancho_red)
                    cuadro[y0:y1, x0:x1] = _a_bytes(tablero_red[y0:y1, x0:x1])

        if resaltar_ultima:
            _marcar_celda(cuadro, int(paso["fila"]), int(paso["columna"]), filas, cols)

        cuadros.append(cuadro)

    return cuadros


def crear_animacion_v2(
    rompecabezas: RompecabezasV2,
    reconstructor,
    ruta_salida: Union[str, Path],
    escala: int = 2,
    cuadros_por_segundo: int = 8,
    cuadros_finales: int = 12,
    titulo: str = "Reconstrucción del Rompecabezas",
    verbose: bool = True,
) -> Path:
    historial = reconstructor if isinstance(reconstructor, list) else getattr(reconstructor, "historial_colocaciones", [])
    if not historial:
        raise ValueError("El historial está vacío. Debe ejecutarse reconstructor.reconstruir() antes de animar.")

    tableros = renderizar_pasos_v2(rompecabezas, historial, escala=escala)

    total = len(historial)
    cuadros = []

    for idx, tablero in enumerate(tableros):
        if idx == 0:
            texto_inf = "Inicio: Tablero vacío"
        else:
            paso = historial[idx - 1]
            p_id = int(paso["id_pieza"])
            f, c = int(paso["fila"]), int(paso["columna"])
            costo = float(paso.get("costo", 0.0))
            texto_inf = f"Paso {idx:2d}: Pieza {p_id:3d} -> ({f}, {c})  costo: {costo:.4f}"

        img_frame = _componer_cuadro(
            tablero_rgb=tablero,
            texto_superior=f"{titulo} [{idx}/{total}]",
            texto_inferior=texto_inf,
            progreso=idx / total if total else 0.0,
        )
        cuadros.append(img_frame)

    cuadros.extend([cuadros[-1]] * cuadros_finales)

    destino = Path(ruta_salida)
    destino.parent.mkdir(parents=True, exist_ok=True)
    duracion = int(1000 / cuadros_por_segundo)

    cuadros[0].save(
        str(destino),
        save_all=True,
        append_images=cuadros[1:],
        duration=duracion,
        loop=0,
        optimize=True,
    )

    if verbose:
        print(f"[Animación] Guardada exitosamente en '{destino}' ({len(cuadros)} fotogramas a {cuadros_por_segundo} fps).")

    return destino
