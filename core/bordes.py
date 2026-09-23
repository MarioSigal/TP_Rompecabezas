"""
Módulo de Extracción de Bordes y Construcción de Matrices de Afinidad.
Proporciona:
- `extraer_banda_borde`: Extrae franjas perimetrales de píxeles orientadas de afuera hacia adentro.
- `compatibilidad_baseline`: Métrica elemental de diferencia absoluta media (L1) para el nivel 1.
- `construir_matrices_afinidad`: Genera las matrices de costo horizontal y vertical entre todas las piezas
  utilizando CUALQUIER función de compatibilidad suministrada por los alumnos.
"""

from typing import Callable, Dict, List, Optional
import numpy as np
from skimage.filters import sobel_v, sobel_h
from skimage.color import rgb2ycbcr
from core.detector_forma import _SHAPE_CACHE
from core.RompecabezasV2 import Pieza

__all__ = [
    "LADOS",
    "BORDES_ENFRENTADOS",
    "extraer_banda_borde",
    "compatibilidad_baseline",
    "limpiar_cache",
    "construir_matrices_afinidad",
]

LADOS = ("NORTE", "SUR", "ESTE", "OESTE")

BORDES_ENFRENTADOS = {
    "horizontal": ("ESTE", "OESTE"),  # B a la derecha de A
    "vertical": ("SUR", "NORTE"),      # B abajo de A
}


def extraer_banda_borde(
    pieza: np.ndarray,
    lado: str,
    cantidad_lineas: int = 1,
    canal: Optional[int] = None,
) -> np.ndarray:
    """
    Devuelve las 'cantidad_lineas' líneas de píxeles del borde indicado.

    Orientación estándar: De AFUERA hacia ADENTRO.
    - Índice 0: Línea exterior más perimétrica.
    - El primer eje siempre corresponde a la longitud del borde (sin importar la orientación),
      lo que permite comparar bordes horizontales y verticales de manera idéntica.
    """
    if lado not in LADOS:
        raise ValueError(f"Lado inválido: '{lado}'. Se espera uno de {LADOS}.")

    alto_px, ancho_px = pieza.shape[:2]
    limite = ancho_px if lado in ("ESTE", "OESTE") else alto_px

    if not 1 <= cantidad_lineas <= limite:
        raise ValueError(
            f"cantidad_lineas={cantidad_lineas} fuera de rango para una pieza "
            f"de {ancho_px}x{alto_px}px en el lado '{lado}'."
        )

    if lado == "OESTE":
        banda = pieza[:, :cantidad_lineas, ...]
    elif lado == "ESTE":
        banda = pieza[:, -cantidad_lineas:, ...][:, ::-1, ...]
    elif lado == "NORTE":
        banda = pieza[:cantidad_lineas, :, ...].swapaxes(0, 1)
    elif lado == "SUR":
        banda = pieza[-cantidad_lineas:, :, ...][::-1, ...].swapaxes(0, 1)

    banda = np.asarray(banda, dtype=np.float64)

    if canal is not None and banda.ndim == 3:
        banda = banda[:, :, canal]

    return banda

def compatibilidad_baseline(
    pieza_a: np.ndarray,
    pieza_b: np.ndarray,
    relacion: str,
) -> float:
    """
    Calcula el costo de acople entre dos piezas usando el error cuadratico medio (Baseline).
    El cuadrado (en vez de valor absoluto) penaliza mucho mas fuerte los saltos grandes y
    casi no penaliza los chicos, lo que agudiza la separacion entre vecinos verdaderos
    (saltos chicos) y falsos (saltos grandes).
    Cuanto MENOR sea el resultado, mayor es la similitud de los bordes.

    Parámetros:
    pieza_a: np.ndarray
        Pieza base (origen).
    pieza_b: np.ndarray
        Pieza vecina propuesta.
    relacion: str
        'horizontal' (B a la derecha de A) o 'vertical' (B abajo de A).

    Retorna:
    float
        Valor de error/costo promedio entre las líneas externas de los bordes.

    Ejemplo de uso
    --------------
    >>> pieza_1 = np.ones((30, 30, 3)) * 100
    >>> pieza_2 = np.ones((30, 30, 3)) * 105
    >>> costo = compatibilidad_baseline(pieza_1, pieza_2, relacion="horizontal")
    >>> print(f"Costo de poner pieza_2 a la derecha de pieza_1: {costo:.1f}")
    Costo de poner pieza_2 a la derecha de pieza_1: 250.0
    """
    if relacion not in BORDES_ENFRENTADOS:
        raise ValueError(f"Relación inválida: '{relacion}'. Se espera 'horizontal' o 'vertical'.")

    lado_a, lado_b = BORDES_ENFRENTADOS[relacion]
    banda_a = extraer_banda_borde(pieza_a, lado_a, cantidad_lineas=1)
    banda_b = extraer_banda_borde(pieza_b, lado_b, cantidad_lineas=1)

    if banda_a.shape[0] != banda_b.shape[0]:
        n_samples = min(banda_a.shape[0], banda_b.shape[0])
        idx_a = np.linspace(0, banda_a.shape[0] - 1, n_samples).astype(int)
        idx_b = np.linspace(0, banda_b.shape[0] - 1, n_samples).astype(int)
        banda_a = banda_a[idx_a]
        banda_b = banda_b[idx_b]

    #Error cuadratico en vez de absoluto: penaliza mucho mas fuerte los saltos
    #grandes y casi no penaliza los chicos, lo que agudiza la separacion entre
    #vecinos verdaderos (saltos chicos) y falsos (saltos grandes)
    return float(np.linalg.norm(banda_a - banda_b)**2)

def limpiar_cache():
    global _SHAPE_CACHE
    _SHAPE_CACHE.clear()

def construir_matrices_afinidad(
    piezas: List[np.ndarray],
    funcion_compatibilidad: Callable[[np.ndarray, np.ndarray, str], float] = compatibilidad_baseline,
) -> Dict[str, np.ndarray]:
    """
    Construye las matrices de costo para todos los pares ordenados de piezas
    usando la función de compatibilidad suministrada.

    Retorna:
        dict: {
            'horizontal': ndarray (N, N) donde matriz[a, b] es el costo de poner B a la derecha de A,
            'vertical': ndarray (N, N) donde matriz[a, b] es el costo de poner B abajo de A
        }
        La diagonal siempre contiene np.inf.
    """
    #Limpiamos el cache de gaussianas de compatibilidad_kl antes de arrancar:
    #evita que una corrida anterior con OTRAS piezas deje entradas cuyo id()
    #de objeto python se haya reciclado para alguna pieza de esta lista (ver
    #_ajustar_gaussiana_borde). Barato si el cache no aplica a la funcion usada.
    limpiar_cache()

    cantidad_piezas = len(piezas)
    matriz_horizontal = np.zeros((cantidad_piezas, cantidad_piezas), dtype=np.float64)
    matriz_vertical = np.zeros((cantidad_piezas, cantidad_piezas), dtype=np.float64)

    for i in range(cantidad_piezas):
        pieza_a = piezas[i]
        for j in range(cantidad_piezas):
            if i == j:
                continue
            pieza_b = piezas[j]

            costo_h = funcion_compatibilidad(pieza_a, pieza_b, "horizontal")
            costo_v = funcion_compatibilidad(pieza_a, pieza_b, "vertical")

            matriz_horizontal[i, j] = costo_h
            matriz_vertical[i, j] = costo_v

    np.fill_diagonal(matriz_horizontal, np.inf)
    np.fill_diagonal(matriz_vertical, np.inf)

    return {
        "horizontal": matriz_horizontal,
        "vertical": matriz_vertical,
    }

def compatibilidad_baseline_por_pieza(
    pieza_a: Pieza,
    pieza_b: Pieza,
    relacion: str,
) -> float:
    if relacion not in BORDES_ENFRENTADOS:
        raise ValueError(f"Relación inválida: '{relacion}'. Se espera 'horizontal' o 'vertical'.")

    lado_a, lado_b = BORDES_ENFRENTADOS[relacion]
    banda_a = pieza_a.calcular_borde_color(lado_a)
    banda_b = pieza_b.calcular_borde_color(lado_b)

    if banda_a.shape[0] != banda_b.shape[0]:
        n_samples = min(banda_a.shape[0], banda_b.shape[0])
        idx_a = np.linspace(0, banda_a.shape[0] - 1, n_samples).astype(int)
        idx_b = np.linspace(0, banda_b.shape[0] - 1, n_samples).astype(int)
        banda_a = banda_a[idx_a]
        banda_b = banda_b[idx_b]

    return float(np.linalg.norm(banda_a - banda_b)**2)


def construir_matrices_afinidad_desde_piezas(
    piezas: List[Pieza],
    funcion_compatibilidad: Callable[[Pieza, Pieza, str], float] = compatibilidad_baseline_por_pieza,
) -> Dict[str, np.ndarray]:
    """
    Construye las matrices de costo para todos los pares ordenados de piezas
    usando la función de compatibilidad suministrada.

    Retorna:
        dict: {
            'horizontal': ndarray (N, N) donde matriz[a, b] es el costo de poner B a la derecha de A,
            'vertical': ndarray (N, N) donde matriz[a, b] es el costo de poner B abajo de A
        }
        La diagonal siempre contiene np.inf.
    """

    cantidad_piezas = len(piezas)
    matriz_horizontal = np.zeros((cantidad_piezas, cantidad_piezas), dtype=np.float64)
    matriz_vertical = np.zeros((cantidad_piezas, cantidad_piezas), dtype=np.float64)

    for i in range(cantidad_piezas):
        pieza_a = piezas[i]
        for j in range(cantidad_piezas):
            if i == j:
                continue
            pieza_b = piezas[j]

            costo_h = funcion_compatibilidad(pieza_a, pieza_b, "horizontal")
            costo_v = funcion_compatibilidad(pieza_a, pieza_b, "vertical")

            matriz_horizontal[i, j] = costo_h
            matriz_vertical[i, j] = costo_v

    np.fill_diagonal(matriz_horizontal, np.inf)
    np.fill_diagonal(matriz_vertical, np.inf)

    return {
        "horizontal": matriz_horizontal,
        "vertical": matriz_vertical,
    }
