"""
Módulo de Extracción y Matching Morfológico de Contornos y Encastres (Nivel 4).
Permite:
1. Binarizar piezas sobre fondo negro puro (0, 0, 0).
2. Segmentar el contorno en sus 4 lados (NORTE, ESTE, SUR, OESTE).
3. Clasificar cada borde en PLANO, SALIENTE (pestaña) o ENTRANTE (muesca/hendidura).
4. Medir la correlación de forma y compatibilidad de encastres.
"""

from typing import Dict, List, Tuple, Any, Optional
from core.geometria_jigsaw import MINIMO_CONTENIDO_MASCARA_FLOAT, MINIMO_CONTENIDO_MASCARA_INT
import numpy as np
from skimage.morphology import binary_erosion, binary_dilation, binary_closing, disk
import cv2

__all__ = [
    "generar_mascara_de_pieza",
    "extraer_contorno_externo",
    "detectar_esquinas_de_pieza",
    "detect_corners_and_split_sides",
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

def generar_mascara_de_pieza(pieza: np.ndarray) -> np.ndarray:
    """
    Binariza la imagen, supone que se sigue el invariante de la mascara.
    """
    if issubclass(pieza.dtype.type, np.floating):
        umbral_invariante_mascara_contenido = MINIMO_CONTENIDO_MASCARA_FLOAT
    else:
        umbral_invariante_mascara_contenido = MINIMO_CONTENIDO_MASCARA_INT
 
    if pieza.ndim == 3:
        canal_maximo = np.max(pieza, axis=2)
    else:
        canal_maximo = pieza
 
    binary = (canal_maximo >= umbral_invariante_mascara_contenido).astype(np.uint8) * 255
    return binary

# mantenemos un alias a las funciones originales para que no se rompa todo en caso de que alguien l a este usando
binarize_piece = generar_mascara_de_pieza

def extraer_contorno_externo(mascara_binaria: np.ndarray) -> np.ndarray:
    """Extrae el contorno exterior principal de mayor área."""

    # Devuelve una cadena de puntos que son el contorno
    contorno, _ = cv2.findContours(mascara_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contorno:
        raise ValueError("No se detectó ningún contorno en la máscara binaria provista.")

    # Deja solo el contorno verdadero
    contorno_filtrado = max(contorno, key=cv2.contourArea)

    # Eliminamos la tercera dimension que solo tiene un canal.
    # Dejamos a la mascara con 2 dimensiones
    return contorno_filtrado.squeeze(axis=1)

extract_external_contour = extraer_contorno_externo

# region encontrar_esquinas

def _punto_en_mitad_izquierda(punto, centro_silueta_x):
    return punto[0] <= centro_silueta_x

def _punto_en_mitad_derecha(punto, centro_silueta_x):
    return punto[0] >= centro_silueta_x

def _punto_en_mitad_superior(punto, centro_silueta_y):
    return punto[1] <= centro_silueta_y

def _punto_en_mitad_inferior(punto, centro_silueta_y):
    return punto[1] >= centro_silueta_y

def _punto_a_la_izquierda_y_por_encima(punto, centro_silueta_x, centro_silueta_y):
    return _punto_en_mitad_izquierda(punto, centro_silueta_x) and _punto_en_mitad_superior(punto, centro_silueta_y)

def _punto_a_la_derecha_y_por_encima(punto, centro_silueta_x, centro_silueta_y):
    return _punto_en_mitad_derecha(punto, centro_silueta_x) and _punto_en_mitad_superior(punto, centro_silueta_y)

def _punto_a_la_izquierda_y_por_debajo(punto, centro_silueta_x, centro_silueta_y):
    return _punto_en_mitad_izquierda(punto, centro_silueta_x) and _punto_en_mitad_inferior(punto, centro_silueta_y)

def _punto_a_la_derecha_y_por_debajo(punto, centro_silueta_x, centro_silueta_y):
    return _punto_en_mitad_derecha(punto, centro_silueta_x) and _punto_en_mitad_inferior(punto, centro_silueta_y)

def _distacia_a_punto(punto, punto_referencia):
    return (punto[0] - punto_referencia[0]) ** 2 + (punto[1] - punto_referencia[1]) ** 2

def _encontrar_punto_mas_cercano_a_otro(puntos, punto_referencia):
    return min(puntos, key=lambda punto: _distacia_a_punto(punto, punto_referencia))

def _encontrar_punto_esquina(puntos_contorno, filtro_cuadrante, coordenada_esquina_absoluta):
    mid_x, mid_y = np.mean(puntos_contorno, axis=0)
    puntos_cuadrante = [p for p in puntos_contorno if filtro_cuadrante(p, mid_x, mid_y)]
    return _encontrar_punto_mas_cercano_a_otro(puntos_cuadrante, coordenada_esquina_absoluta)

def detectar_esquinas_de_pieza(contour_pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Detecta las 4 esquinas base (TL, TR, BR, BL) de la pieza."""
    """Supone una pieza, balanceada, sino se rompe, ver detectar_esquinas_de_pieza_desde_mascara"""

    x_inicio, x_fin = np.min(contour_pts[:, 0]), np.max(contour_pts[:, 0])
    y_inicio, y_fin = np.min(contour_pts[:, 1]), np.max(contour_pts[:, 1])

    esquina_top_left_absoluta = (x_inicio, y_inicio)
    esquina_top_right_absoluta = (x_fin, y_inicio)
    esquina_bottom_right_absoluta = (x_fin, y_fin)
    esquina_bottom_left_absoluta = (x_inicio, y_fin)

    esquina_top_left = _encontrar_punto_esquina(contour_pts, _punto_a_la_izquierda_y_por_encima, esquina_top_left_absoluta)
    esquina_top_right = _encontrar_punto_esquina(contour_pts, _punto_a_la_derecha_y_por_encima, esquina_top_right_absoluta)
    esquina_bottom_right = _encontrar_punto_esquina(contour_pts, _punto_a_la_derecha_y_por_debajo, esquina_bottom_right_absoluta)
    esquina_bottom_left = _encontrar_punto_esquina(contour_pts, _punto_a_la_izquierda_y_por_debajo, esquina_bottom_left_absoluta)
 
    return np.array(esquina_top_left), np.array(esquina_top_right), np.array(esquina_bottom_right), np.array(esquina_bottom_left)

def encontrar_gaps_en_arreglo(arreglo: np.ndarray):
    # Find lower and upper bounds of the gaps
    lower_bounds = (arreglo + 1)[:-1]
    upper_bounds = (arreglo - 1)[1:]

    # Filter valid gap ranges where lower <= upper
    mask = lower_bounds <= upper_bounds
    gap_starts = lower_bounds[mask]
    gap_ends = upper_bounds[mask]
    return list(zip(gap_starts, gap_ends))

def llenar_gaps_verticales(mascara:np.ndarray):

    for fila in mascara:
        indices_mascara = np.where(fila) 
        gaps = encontrar_gaps_en_arreglo(indices_mascara[0])
        for inicio, fin in gaps:
            fila[inicio: fin + 1] = True

    return mascara

def llenar_gaps_horizontales(mascara:np.ndarray):
    return llenar_gaps_verticales(mascara.T).T

def detectar_esquinas_de_pieza_desde_mascara(mascara: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Detecta las 4 esquinas base (TL, TR, BR, BL) de la pieza."""

    coordenadas_y, coordenadas_x = np.where(mascara)
    y_inicio, y_final = np.min(coordenadas_y), np.max(coordenadas_y)
    x_inicio, x_final = np.min(coordenadas_x), np.max(coordenadas_x)
    
    altura = y_final - y_inicio  
    ancho = x_final - x_inicio  

    mascara_cerrada = llenar_gaps_verticales(mascara.copy())
    mascara_cerrada = llenar_gaps_horizontales(mascara_cerrada)    

    # no hacemos la mitad por si una muesca es muy profunda y llega a cruzar toda la pieza
    kernel_vertical = np.ones((int(altura//1.75), 1))
    kernel_horizontal = np.ones((1, int(ancho//1.75)))

    mascara_fondo = ~mascara_cerrada
    fondo_dilatado_verticalmente = binary_dilation(mascara_fondo, kernel_vertical)
    fondo_con_muescas_horizontales = binary_erosion(fondo_dilatado_verticalmente, kernel_vertical)
    fondo_dilatado_horizontalmente = binary_dilation(fondo_con_muescas_horizontales, kernel_horizontal)
    fondo_con_muescas = binary_erosion(fondo_dilatado_horizontalmente, kernel_horizontal)

    mascara_rectangulo = ~fondo_con_muescas
    mascara_binaria = (mascara_rectangulo).astype(np.uint8) * 255
    try:
        contorno_simetrico = extraer_contorno_externo(mascara_binaria)
    except ValueError:
        mascara_original = (mascara).astype(np.uint8) * 255
        # Fallback si la pieza no estaba recta
        contorno_simetrico = extraer_contorno_externo(mascara_original)

    return detectar_esquinas_de_pieza(contorno_simetrico)

detect_jigsaw_corners = detectar_esquinas_de_pieza

# endregion

def pasar_borde_a_1d(
    puntos_contorno_borde: np.ndarray,
    nombre_lado: str = "NORTE",
) -> Dict[str, Any]:
    """
    Proyecta una curva 2D de contorno respecto a la recta que une sus extremos
    y extrae la señal 1D de desviación perpendicular.
    Clasifica el tipo de borde en: 'PLANO', 'SALIENTE' o 'ENTRANTE'.
    """
    cantidad_de_puntos = len(puntos_contorno_borde)

    punto_inicio_borde = puntos_contorno_borde[0].astype(np.float32)
    punto_final_borde = puntos_contorno_borde[-1].astype(np.float32)
    
    distancia = np.linalg.norm(punto_final_borde - punto_inicio_borde)
    if distancia == 0:
        return {
            "type": "PLANO",
            "profile": np.zeros(cantidad_de_puntos, dtype=np.float32),
            "norm": 0.0,
            "length": 0.0,
            "max_dev": 0.0,
            "mean_dev": 0.0,
        }

    # El del norte es contrario porque cuando baja es mas grande, lo mismo oeste para la derecha.
    lado_u = nombre_lado.upper()
    if lado_u == "NORTE":
        eje_de_movimento = np.array([0.0, -1.0])
    elif lado_u == "SUR":
        eje_de_movimento = np.array([0.0, 1.0])
    elif lado_u == "OESTE":
        eje_de_movimento = np.array([-1.0, 0.0])
    elif lado_u == "ESTE":
        eje_de_movimento = np.array([1.0, 0.0])
    else:
        raise ValueError("Nombre de Borde no Valido")
    
    silueta_de_borde = puntos_contorno_borde.astype(np.float32) - punto_inicio_borde
    cambio_profundidad_a_lo_largo_del_borde = np.dot(silueta_de_borde, eje_de_movimento)
 
    maximo_cambio = np.max(np.abs(cambio_profundidad_a_lo_largo_del_borde))
    desviacion_promedio = float(np.mean(cambio_profundidad_a_lo_largo_del_borde))
 
    if maximo_cambio < distancia * 0.06:
        tipo_de_borde = "PLANO"
        cambio_profundidad_a_lo_largo_del_borde = np.zeros(cantidad_de_puntos, dtype=np.float32)
    elif desviacion_promedio > 0:
        tipo_de_borde = "SALIENTE"
    else:
        tipo_de_borde = "ENTRANTE"
 
    norma_cambio_profundidad = float(np.linalg.norm(cambio_profundidad_a_lo_largo_del_borde))
    return {
        "type": tipo_de_borde,
        "profile": cambio_profundidad_a_lo_largo_del_borde,
        "norm": norma_cambio_profundidad,
        "length": distancia,
        "max_dev": maximo_cambio,
        "mean_dev": desviacion_promedio,
    }
 
 
extraer_perfil_1d = pasar_borde_a_1d


def _muestrear_color_curva(
    puntos_de_borde: np.ndarray,
    imagen_rgb: np.ndarray,
) -> np.ndarray:
    """
    Extrae el color RGB de `imagen_rgb` en el borde, 
    """
    coordenadas_x = puntos_de_borde[:, 0]
    coordenadas_y = puntos_de_borde[:, 1]
 
    ccolor_arr = imagen_rgb[coordenadas_y, coordenadas_x].astype(np.float32)
    return ccolor_arr

 
def encontra_indice_de_punto_en_arreglo_puntos(puntos, punto):
    d = np.sum((puntos - punto) ** 2, axis=1)
    return int(np.argmin(d))

def recoger_border_recorriendo_entre_puntos(contorno, idx_primer_esquina, idx_segunda_esquina, idx_tercer_esquina):

    primer_punto = contorno[0:1]

    borde_1 = contorno[0 : idx_primer_esquina + 1]
    borde_2 = contorno[idx_primer_esquina : idx_segunda_esquina + 1]
    borde_3 = contorno[idx_segunda_esquina : idx_tercer_esquina + 1][::-1]

    borde_4_sin_inicio = contorno[idx_tercer_esquina:] 
    borde_4 = np.vstack([borde_4_sin_inicio, primer_punto])
    borde_4 = borde_4[::-1]

    return borde_1, borde_2, borde_3, borde_4

def detect_corners_and_split_sides(
    contour_pts: np.ndarray,
    binary_mask: np.ndarray,
    imagen_rgb: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Segmenta el contorno en los 4 lados orientados (NORTE, ESTE, SUR, OESTE)
    y calcula la señal 1D de desviación perpendicular.
    """

    binary_mask_booleana = binary_mask > 0
    esquina_top_left, esquina_top_right, esquina_bottom_right, esquina_bottom_left = detectar_esquinas_de_pieza_desde_mascara(binary_mask_booleana)
 
    idx_top_left = encontra_indice_de_punto_en_arreglo_puntos(contour_pts, esquina_top_left)
    
    # Suponemos que la lista sigue un orden entre punto a punto
    # Centramos la lista empezando desde la esquina superior izquierda
    contorno_desde_esquina_top_left = np.roll(contour_pts, -idx_top_left, axis=0)

    idx_top_right = encontra_indice_de_punto_en_arreglo_puntos(contorno_desde_esquina_top_left, esquina_top_right)
    idx_bottom_right = encontra_indice_de_punto_en_arreglo_puntos(contorno_desde_esquina_top_left, esquina_bottom_right)
    idx_bottom_left = encontra_indice_de_punto_en_arreglo_puntos(contorno_desde_esquina_top_left, esquina_bottom_left)
 
    if idx_bottom_left < idx_top_right:
        # Recorrido antihorario
        bordes = recoger_border_recorriendo_entre_puntos(contorno_desde_esquina_top_left, idx_bottom_left, idx_bottom_right, idx_top_right)
        silueta_borde_oeste = bordes[0]
        silueta_borde_sur = bordes[1]
        silueta_borde_este = bordes[2]
        silueta_borde_norte = bordes[3]
    else:
        # Recorrido horario
        bordes = recoger_border_recorriendo_entre_puntos(contorno_desde_esquina_top_left, idx_top_right, idx_bottom_right, idx_bottom_left)
        silueta_borde_norte = bordes[0]
        silueta_borde_este = bordes[1]
        silueta_borde_sur = bordes[2]
        silueta_borde_oeste = bordes[3]
 
    informacion_silueta_norte = pasar_borde_a_1d(silueta_borde_norte, "NORTE")
    informacion_silueta_este = pasar_borde_a_1d(silueta_borde_este, "ESTE")
    informacion_silueta_sur = pasar_borde_a_1d(silueta_borde_sur, "SUR")
    informacion_silueta_oeste = pasar_borde_a_1d(silueta_borde_oeste, "OESTE")

    descriptor_borde_norte = (silueta_borde_norte, informacion_silueta_norte) 
    descriptor_borde_este = (silueta_borde_este, informacion_silueta_este)
    descriptor_borde_sur = (silueta_borde_sur, informacion_silueta_sur)
    descriptor_borde_oeste = (silueta_borde_oeste, informacion_silueta_oeste)
 
    #Perfil de color a lo largo de cada costura
    #Sigue la CURVA del encastre, no una columna recta: por eso hace falta la imagen ademas del contorno.
    if imagen_rgb is not None:
        for curva, informacion in [descriptor_borde_norte, descriptor_borde_este, descriptor_borde_sur, descriptor_borde_oeste]:
            colores = _muestrear_color_curva(curva, imagen_rgb)
            informacion["color_profile"] = colores
            informacion["curve"] = curva
 
    lista_tipos_bordes = [informacion_silueta_norte["type"], informacion_silueta_este["type"], informacion_silueta_sur["type"], informacion_silueta_oeste["type"]]
    cantidad_bordes_planos = sum(1 for tipo_borde in lista_tipos_bordes if tipo_borde == "PLANO")

    if cantidad_bordes_planos == 2:
        tipo_pieza = "ESQUINA"
    elif cantidad_bordes_planos == 1:
        tipo_pieza = "LADO"
    else:
        tipo_pieza = "INTERIOR"
 
    return {
        "NORTE": informacion_silueta_norte,
        "ESTE": informacion_silueta_este,
        "SUR": informacion_silueta_sur,
        "OESTE": informacion_silueta_oeste,
        "tipo_pieza": tipo_pieza,
        "num_flat": cantidad_bordes_planos,
        "num_planos": cantidad_bordes_planos,
        "corners": {"TL": esquina_top_left, "TR": esquina_top_right, "BR": esquina_bottom_right, "BL": esquina_bottom_left},
    }

# Caché para no recalcular la forma de la misma pieza repetidamente
_SHAPE_CACHE = {}

def get_cache_id(pieza: np.ndarray):
    # hasheamos los bytes del array
    originally_writeable = pieza.flags.writeable
    try:
        pieza.flags.writeable = False
        return hash((pieza.data.tobytes(), pieza.shape, pieza.dtype))
    finally:
        pieza.flags.writeable = originally_writeable

def analyze_piece_shape(pieza: np.ndarray) -> Dict[str, Any]:
    """Analiza la silueta morfológica de una pieza."""
    global _SHAPE_CACHE

    #Verificamos el cache primero
    cache_id = get_cache_id(pieza)

    if cache_id is not None and cache_id in _SHAPE_CACHE:
        return _SHAPE_CACHE[cache_id]

    if len(_SHAPE_CACHE) > 2000:
        _SHAPE_CACHE.clear()

    binary = generar_mascara_de_pieza(pieza)
    contour = extraer_contorno_externo(binary)
    sides_info = detect_corners_and_split_sides(contour, binary, imagen_rgb=pieza)

    resultado = {
        "binary_mask": binary,
        "contour": contour,
        "sides": sides_info,
        "topology": sides_info["tipo_pieza"],
    }

    if cache_id is not None:
        _SHAPE_CACHE[cache_id] = resultado

    return resultado

#Alias en español para el trabajo práctico de los estudiantes
segmentar_borde_en_4 = analyze_piece_shape
analizar_silueta_de_pieza = analyze_piece_shape 

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

def indices_interpolacion_entre_dos_bordes(silueta_a, silueta_b):
    n_samples = min(silueta_a.shape[0], silueta_b.shape[0])
    idx_a = np.linspace(0, silueta_a.shape[0] - 1, n_samples).astype(int)
    idx_b = np.linspace(0, silueta_b.shape[0] - 1, n_samples).astype(int)

    return idx_a, idx_b

def interpolar_valores_bordes(valores_silueta_a, valores_silueta_b):
    if valores_silueta_a.shape[0] != valores_silueta_b.shape[0]:
        idx_a, idx_b = indices_interpolacion_entre_dos_bordes(valores_silueta_a, valores_silueta_b)
        valores_a_interpolados = valores_silueta_a[idx_a]
        valores_b_interpolados = valores_silueta_b[idx_b]
    else:
        valores_a_interpolados = valores_silueta_a
        valores_b_interpolados = valores_silueta_b

    return valores_a_interpolados, valores_b_interpolados 

def calcular_mse_color_bordes(informacion_borde_a: Dict[str, Any], informacion_borde_b: Dict[str, Any]) -> float:
    """
    Calcula el Error Cuadrático Medio (MSE) entre los perfiles de color de dos bordes enfrentados.
    Los perfiles corren orientados en el mismo sentido a lo largo de la costura.
    """
    color_a = informacion_borde_a.get("color_profile")
    color_b = informacion_borde_b.get("color_profile")
    if color_a is None or color_b is None:
        return 0.0

    if color_a.shape[0] != color_b.shape[0]:
        idx_a, idx_b = indices_interpolacion_entre_dos_bordes(color_a, color_b)
        color_a = color_a[idx_a]
        color_b = color_b[idx_b]

    return float(np.mean((color_a - color_b)**2))

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

