"""
Módulo de Animación del Proceso de Reconstrucción.
Genera un GIF animado que muestra al algoritmo armando el rompecabezas paso a paso
según el `historial_colocaciones` del reconstructor.
"""

from pathlib import Path
from typing import Union, List, Dict, Any
import numpy as np
from PIL import Image, ImageDraw
from typing import Optional

__all__ = ["crear_animacion", "renderizar_pasos"]

COLOR_FONDO = (18, 22, 28)
COLOR_CELDA_VACIA = (32, 38, 46)
COLOR_TEXTO = (205, 220, 235)
COLOR_TEXTO_TENUE = (130, 148, 165)
COLOR_BARRA = (0, 190, 130)
COLOR_BARRA_FONDO = (38, 46, 56)
COLOR_RESALTADO = (255, 190, 60)


def _a_bytes(imagen_float: np.ndarray) -> np.ndarray:
    return (np.clip(imagen_float, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def _escalar_por_bloques(imagen: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return imagen

    alto, ancho = imagen.shape[:2]
    alto_util = (alto // factor) * factor
    ancho_util = (ancho // factor) * factor
    recorte = imagen[:alto_util, :ancho_util]

    nueva_forma = (alto_util // factor, factor, ancho_util // factor, factor) + imagen.shape[2:]
    return recorte.reshape(nueva_forma).mean(axis=(1, 3))


def _dibujar_grilla_vacia(alto_px: int, ancho_px: int, cantidad_filas: int, cantidad_columnas: int) -> np.ndarray:
    lienzo = np.full((alto_px, ancho_px, 3), COLOR_CELDA_VACIA, dtype=np.uint8)

    alto_celda = alto_px / cantidad_filas
    ancho_celda = ancho_px / cantidad_columnas

    for i in range(cantidad_filas + 1):
        y = min(int(round(i * alto_celda)), alto_px - 1)
        lienzo[y, :] = COLOR_FONDO

    for j in range(cantidad_columnas + 1):
        x = min(int(round(j * ancho_celda)), ancho_px - 1)
        lienzo[:, x] = COLOR_FONDO

    return lienzo


def _componer_cuadro(
    tablero_rgb: np.ndarray,
    texto_superior: str,
    texto_inferior: str,
    progreso: float,
    alto_hud: int = 34,
    alto_pie: int = 24,
) -> Image.Image:
    alto_tablero, ancho_tablero = tablero_rgb.shape[:2]
    ancho_cuadro = max(ancho_tablero, 400)
    alto_cuadro = alto_tablero + alto_hud + alto_pie

    cuadro = np.full((alto_cuadro, ancho_cuadro, 3), COLOR_FONDO, dtype=np.uint8)
    desplazamiento_x = (ancho_cuadro - ancho_tablero) // 2
    cuadro[alto_hud : alto_hud + alto_tablero, desplazamiento_x : desplazamiento_x + ancho_tablero] = tablero_rgb

    y_barra = alto_hud - 5
    cuadro[y_barra : y_barra + 3, :] = COLOR_BARRA_FONDO
    ancho_avance = int(round(progreso * ancho_cuadro))
    if ancho_avance > 0:
        cuadro[y_barra : y_barra + 3, :ancho_avance] = COLOR_BARRA

    imagen = Image.fromarray(cuadro)
    lapiz = ImageDraw.Draw(imagen)
    lapiz.text((10, 8), texto_superior, fill=COLOR_TEXTO)
    lapiz.text((10, alto_hud + alto_tablero + 7), texto_inferior, fill=COLOR_TEXTO_TENUE)

    return imagen


def _marcar_celda(cuadro: np.ndarray, fila: int, columna: int, cantidad_filas: int, cantidad_columnas: int, grosor: int = 2) -> None:
    alto_px, ancho_px = cuadro.shape[:2]
    alto_celda = alto_px / cantidad_filas
    ancho_celda = ancho_px / cantidad_columnas

    y0 = int(round(fila * alto_celda))
    y1 = min(int(round((fila + 1) * alto_celda)), alto_px) - 1
    x0 = int(round(columna * ancho_celda))
    x1 = min(int(round((columna + 1) * ancho_celda)), ancho_px) - 1

    cuadro[y0 : y0 + grosor, x0 : x1 + 1] = COLOR_RESALTADO
    cuadro[y1 - grosor + 1 : y1 + 1, x0 : x1 + 1] = COLOR_RESALTADO
    cuadro[y0 : y1 + 1, x0 : x0 + grosor] = COLOR_RESALTADO
    cuadro[y0 : y1 + 1, x1 - grosor + 1 : x1 + 1] = COLOR_RESALTADO


def renderizar_pasos(
    rompecabezas,
    historial: List[Dict[str, Any]],
    escala: int = 2,
    resaltar_ultima: bool = True,
    piezas: Optional[List[np.ndarray]] = None,
) -> List[np.ndarray]:
    filas = rompecabezas.cantidad_filas
    cols = rompecabezas.cantidad_columnas

    tablero_completo = rompecabezas.pegar_piezas([[0] * cols for _ in range(filas)], piezas=piezas)
    tablero_escalado = _escalar_por_bloques(tablero_completo, escala)
    alto_red, ancho_red = tablero_escalado.shape[:2]

    fondo = _dibujar_grilla_vacia(alto_red, ancho_red, filas, cols)
    grilla_parcial = np.full((filas, cols), -1, dtype=int)
    cuadros = [fondo.copy()]

    for paso in historial:
        grilla_parcial[paso["fila"], paso["columna"]] = int(paso["id_pieza"])

        # Generar tablero parcial
        tablero = rompecabezas.pegar_piezas(grilla_parcial, piezas=piezas)
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


def crear_animacion(
    rompecabezas,
    reconstructor,
    ruta_salida: Union[str, Path],
    escala: int = 2,
    cuadros_por_segundo: int = 8,
    cuadros_finales: int = 12,
    titulo: str = "Reconstrucción del Rompecabezas",
    verbose: bool = True,
    piezas: Optional[List[np.ndarray]] = None,
) -> Path:
    historial = reconstructor if isinstance(reconstructor, list) else getattr(reconstructor, "historial_colocaciones", [])
    if not historial:
        raise ValueError("El historial está vacío. Debe ejecutarse reconstructor.reconstruir() antes de animar.")

    tableros = renderizar_pasos(rompecabezas, historial, escala=escala, piezas=piezas)

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
