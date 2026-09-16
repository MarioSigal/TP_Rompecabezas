"""
Módulo de Extracción de Bordes y Construcción de Matrices de Afinidad.
Proporciona:
- `extraer_banda_borde`: Extrae franjas perimetrales de píxeles orientadas de afuera hacia adentro.
- `compatibilidad_baseline`: Métrica elemental de diferencia absoluta media (L1) para el nivel 1.
- `construir_matrices_afinidad`: Genera las matrices de costo horizontal y vertical entre todas las piezas
  utilizando CUALQUIER función de compatibilidad suministrada por los alumnos.
"""

from typing import Callable, Dict, List, Optional, Union, Any
import numpy as np
from skimage.filters import sobel_v, sobel_h
from skimage.color import rgb2ycbcr

__all__ = [
    "LADOS",
    "BORDES_ENFRENTADOS",
    "extraer_banda_borde",
    "compatibilidad_baseline",
    "compatibilidad_gradiante",
    "compatibilidad_kl",
    "compatibilidad_baseline_kl",
    "limpiar_cache_kl",
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
    pieza_a: Union[np.ndarray, Dict[str, Any]],
    pieza_b: Union[np.ndarray, Dict[str, Any]],
    relacion: Optional[str] = None,
) -> float:
    """
    Calcula el costo de acople entre dos piezas o dos bordes usando el error cuadrático medio (MSE).
    Cuanto MENOR sea el resultado, mayor es la similitud de los bordes.

    Soporta múltiples formas de uso:
    1. Perfiles de color o lados directos (ej. para el Nivel 4 y 6 con 'segmentar_borde_en_4'):
       >>> costo = compatibilidad_baseline(lado_a["color_profile"], lado_b["color_profile"])
       >>> costo = compatibilidad_baseline(lado_a, lado_b)
    2. Piezas con encastres curvos (Jigsaw) con relación ('horizontal' o 'vertical'):
       Extrae automáticamente las costuras curvas mediante 'segmentar_borde_en_4'
       y calcula el MSE de los perfiles de color de la costura.
    3. Piezas rectangulares estándar con relación:
       >>> costo = compatibilidad_baseline(pieza_1, pieza_2, relacion="horizontal")
    """
    def _extraer_vector_color(obj):
        if isinstance(obj, dict):
            for k in ("color_profile", "color", "profile"):
                if k in obj and obj[k] is not None:
                    return np.asarray(obj[k], dtype=np.float64)
            raise ValueError(f"El diccionario de lado no contiene 'color_profile' ni 'profile': {list(obj.keys())}")
        return np.asarray(obj, dtype=np.float64)

    # Detectar si pieza_a o pieza_b son vectores/lados directos o si no se pasó relación
    es_lado_o_vector_a = isinstance(pieza_a, dict) or (
        isinstance(pieza_a, np.ndarray) and (pieza_a.ndim <= 2 or (pieza_a.ndim == 3 and min(pieza_a.shape[:2]) <= 3))
    )
    es_lado_o_vector_b = isinstance(pieza_b, dict) or (
        isinstance(pieza_b, np.ndarray) and (pieza_b.ndim <= 2 or (pieza_b.ndim == 3 and min(pieza_b.shape[:2]) <= 3))
    )

    if es_lado_o_vector_a or es_lado_o_vector_b or relacion is None:
        banda_a = _extraer_vector_color(pieza_a)
        banda_b = _extraer_vector_color(pieza_b)
    else:
        if relacion not in BORDES_ENFRENTADOS:
            raise ValueError(f"Relación inválida: '{relacion}'. Se espera 'horizontal' o 'vertical'.")

        lado_a, lado_b = BORDES_ENFRENTADOS[relacion]

        # Detectar si son piezas Jigsaw (con fondo negro o padding en las esquinas)
        es_jigsaw = (
            np.all(pieza_a[0, 0] < 0.01) and np.all(pieza_a[-1, -1] < 0.01)
        ) or (
            np.mean(np.all(pieza_a < 0.005, axis=-1) if pieza_a.ndim == 3 else (pieza_a < 0.005)) > 0.05
        )

        if es_jigsaw:
            try:
                from .detector_forma import segmentar_borde_en_4
            except ImportError:
                from detector_forma import segmentar_borde_en_4

            info_a = segmentar_borde_en_4(pieza_a)
            info_b = segmentar_borde_en_4(pieza_b)
            banda_a = np.asarray(info_a[lado_a]["color_profile"], dtype=np.float64)
            banda_b = np.asarray(info_b[lado_b]["color_profile"], dtype=np.float64)
        else:
            banda_a = extraer_banda_borde(pieza_a, lado_a, cantidad_lineas=1)
            banda_b = extraer_banda_borde(pieza_b, lado_b, cantidad_lineas=1)

    if banda_a.shape[0] != banda_b.shape[0]:
        n_samples = min(banda_a.shape[0], banda_b.shape[0])
        idx_a = np.linspace(0, banda_a.shape[0] - 1, n_samples).astype(int)
        idx_b = np.linspace(0, banda_b.shape[0] - 1, n_samples).astype(int)
        banda_a = banda_a[idx_a]
        banda_b = banda_b[idx_b]

    # Error cuadrático medio por punto
    if banda_a.ndim >= 2 and banda_a.shape[-1] == 3:
        return float(np.mean(np.sum((banda_a - banda_b) ** 2, axis=-1)))
    return float(np.linalg.norm(banda_a - banda_b) ** 2)

def _gradientes_franja(franja):
    """
    Aplica Sobel horizontal Y vertical de skimage sobre una franja de LUMINANCIA
    (largo_del_borde, 3) y devuelve ambos, en su columna central, por fila:

        gradiente_x: derivada A LO LARGO DE LAS COLUMNAS (profundidad, hacia
            adentro/afuera de la pieza). Se usa para EXTRAPOLAR el valor del
            borde a partir del interior.
        gradiente_y: derivada A LO LARGO DE LAS FILAS (a lo largo del borde).
            El gradiente_x solo no detecta si una textura/borde DIAGONAL que
            cruza la costura en angulo se corta de golpe: dos piezas realmente
            vecinas deberian traer una pendiente parecida tambien en esta
            direccion, no solo en la de profundidad.

    Solo nos interesa la columna CENTRAL: es la unica con una ventana 3x3
    completa y real (las columnas 0 y 2 son artefactos de padding de sobel en
    los bordes de esta franja de 3 columnas, no gradientes confiables).
    """
    gradiente_x = sobel_v(franja)[:, 1]
    gradiente_y = sobel_h(franja)[:, 1]
    return gradiente_x, gradiente_y

def compatibilidad_gradiante(pieza_a, pieza_b, relacion):
    """
    Calcula el costo de acople entre dos piezas EXTRAPOLANDO, en X (profundidad)
    unicamente, desde el borde real de cada una hacia el borde real de la otra,
    sobre la LUMINANCIA (canal Y de YCbCr) unicamente.

    A diferencia de versiones anteriores, el borde NO se deja afuera: se usa
    tanto para estimar el gradiente (la ventana de 3 columnas usada para el
    Sobel arranca justo en el borde) como para arrancar la extrapolacion (se
    predice desde el valor real del borde de A, no desde un punto interior mas
    adentro). Como el borde de A esta a un solo paso del borde de B (cruzando
    la costura), solo hace falta extrapolar 1 paso.

    Cuanto mas se parezca esa prediccion al valor real del borde de la otra
    pieza, menor el costo. Se hace en ambos sentidos (A->B y B->A) y se suman.

    Usar solo luminancia (en vez de los 3 canales RGB) hace la comparacion mas
    robusta al ruido de color/cromatico, que no aporta informacion real sobre
    la continuidad de la forma/textura en el borde.
    Cuanto MENOR sea el resultado, mayor es la similitud/continuidad de los bordes.

    Parámetros:
    pieza_a: np.ndarray
        Pieza base (origen).
    pieza_b: np.ndarray
        Pieza vecina propuesta.
    relacion: str
        'horizontal' (B a la derecha de A) o 'vertical' (B abajo de A).

    Retorna:
    float
        Suma, para A hacia B y para B hacia A, del error de prediccion (norma
        al cuadrado, sobre la luminancia).

    Ejemplo de uso
    --------------
    >>> # Usa imagenes en el rango [0, 1] (como cargar_imagen): rgb2ycbcr espera
    >>> # ese rango, a diferencia de compatibilidad_baseline que es lineal y no le importa la escala.
    >>> pieza_1 = np.ones((30, 30, 3)) * 0.30
    >>> pieza_2 = np.ones((30, 30, 3)) * 0.70
    >>> costo = compatibilidad_gradiante(pieza_1, pieza_2, relacion="horizontal")
    >>> print(f"Costo de poner pieza_2 a la derecha de pieza_1: {costo:.1f}")
    Costo de poner pieza_2 a la derecha de pieza_1: 7.1
    """
    if relacion not in BORDES_ENFRENTADOS:
        raise ValueError(
            f"Relación inválida: '{relacion}'. Se espera 'horizontal' o 'vertical'."
        )

    lado_a, lado_b = BORDES_ENFRENTADOS[relacion]

    #3 lineas afuera->adentro (la costura + 2 interiores): la costura queda
    #adentro de la ventana usada para el Sobel, no se descarta como antes
    banda_a = extraer_banda_borde(pieza_a, lado_a, cantidad_lineas=3)
    banda_b = extraer_banda_borde(pieza_b, lado_b, cantidad_lineas=3)

    #Reorientamos A a orden espacial real: indice 0 = mas interno, indice 2 = borde (pegado a B).
    #B ya viene en ese orden: indice 0 = borde (pegado a A), indice 2 = mas interno.
    tira_a = banda_a[:, ::-1, ...]
    tira_b = banda_b

    #Nos quedamos SOLO con la luminancia (canal Y de YCbCr). rgb2ycbcr deja el
    #canal en el ultimo eje (..., 3) -> [..., 0] toma la Y y descarta el eje de
    #canales (no el eje de a lo largo del borde).
    tira_a = rgb2ycbcr(tira_a)[..., 0] / 255
    tira_b = rgb2ycbcr(tira_b)[..., 0] / 255

    #Gradiente centrado en la columna del medio de esta ventana de 3 (la unica
    #con vecinos reales a ambos lados); usamos solo X, no hace falta Y aca.
    gradiente_x_a, _ = _gradientes_franja(tira_a)
    gradiente_x_b, _ = _gradientes_franja(tira_b)

    borde_a = tira_a[:, -1]  # linea de contacto real de A (luminancia)
    borde_b = tira_b[:, 0]   # linea de contacto real de B (luminancia)

    #Extrapolamos 1 solo paso, DESDE el borde real (no desde un punto interior).
    #sobel_v da una derivada con diferencia central de 2 pasos (vale 2x la
    #pendiente real por pixel), por eso se divide por 2 para dar un solo paso.
    #En B el eje de profundidad crece hacia adentro, por eso se resta en vez de sumar.
    prediccion_de_b = borde_a + gradiente_x_a * 0.5
    prediccion_de_a = borde_b - gradiente_x_b * 0.5

    diferencia_b = float(np.linalg.norm(prediccion_de_b - borde_b) ** 2)
    diferencia_a = float(np.linalg.norm(prediccion_de_a - borde_a) ** 2)

    return diferencia_b + diferencia_a

#Piso de regularizacion para las covarianzas de compatibilidad_kl: sin esto, una
#pieza casi plana (varianza local casi nula) da una covarianza casi singular y
#el logaritmo del determinante (y la inversa) se rompen numericamente.
_EPSILON_COVARIANZA_KL = 1e-6

def _muestras_parches_3x3(pieza, lado, profundidad):
    """
    Extrae, cerca del borde indicado, TODOS los parches de 3x3 pixeles (sobre
    LUMINANCIA) como si fueran muestras de una variable aleatoria de 9
    dimensiones. Usa una ventana deslizante: cada parche se solapa con sus
    vecinos, asi que no son independientes entre si, pero alcanza para estimar
    la media y la covarianza local de la pieza.

    Retorna un array (cantidad_de_parches, 9).
    """
    banda = extraer_banda_borde(pieza, lado, cantidad_lineas=profundidad)
    luminancia = rgb2ycbcr(banda)[..., 0] / 255.0
    ventanas = np.lib.stride_tricks.sliding_window_view(luminancia, (3, 3))
    filas, columnas = ventanas.shape[:2]
    return ventanas.reshape(filas * columnas, 9)


#Cache de gaussianas por (pieza, lado, profundidad) para compatibilidad_kl.
#Una pieza tiene solo 4 lados, pero construir_matrices_afinidad la compara
#contra las N-1 restantes: sin cache, el mismo lado de la misma pieza se
#reajustaria (media + covarianza, via sliding_window_view + np.cov) una vez
#por cada comparacion en la que participa, en vez de una sola vez.
#
#Usamos id(pieza) como parte de la clave -> SOLO valido mientras las piezas
#sigan vivas (nunca se liberan y su memoria se reutiliza para otro array). Eso
#es exactamente lo que pasa dentro de una corrida de construir_matrices_afinidad
#(la lista `piezas` las mantiene vivas todo el tiempo). Si se llama a
#compatibilidad_kl / compatibilidad_baseline_kl sueltas en un loop con listas
#de piezas DISTINTAS entre corridas, llamar limpiar_cache_kl() entre una y otra.
_CACHE_GAUSSIANA_BORDE = {}


def limpiar_cache_kl():
    """Vacia el cache de gaussianas de compatibilidad_kl (ver _CACHE_GAUSSIANA_BORDE)."""
    _CACHE_GAUSSIANA_BORDE.clear()


def _ajustar_gaussiana_borde(pieza, lado, profundidad):
    """
    Media y covarianza de los parches de 3x3 cerca del borde indicado,
    cacheadas por (pieza, lado, profundidad): la pieza no cambia entre una
    comparacion y la siguiente, asi que calcular esto una sola vez por pieza
    (en vez de una vez por CADA comparacion en la que participa) alcanza.
    """
    clave = (id(pieza), lado, profundidad)
    if clave in _CACHE_GAUSSIANA_BORDE:
        return _CACHE_GAUSSIANA_BORDE[clave]

    muestras = _muestras_parches_3x3(pieza, lado, profundidad)
    media = muestras.mean(axis=0)
    covarianza = np.cov(muestras, rowvar=False)

    _CACHE_GAUSSIANA_BORDE[clave] = (media, covarianza)
    return media, covarianza


def _kl_gaussianas_multivariadas(media_p, covarianza_p, media_q, covarianza_q):
    """
    KL(P || Q) entre dos gaussianas multivariadas, formula cerrada:
        0.5 * [tr(Σq⁻¹Σp) + (μq-μp)ᵀΣq⁻¹(μq-μp) - k + ln(|Σq| / |Σp|)]
    """
    dimensiones = media_p.shape[0]
    identidad = np.eye(dimensiones)
    covarianza_q_regularizada = covarianza_q + _EPSILON_COVARIANZA_KL * identidad
    covarianza_p_regularizada = covarianza_p + _EPSILON_COVARIANZA_KL * identidad
    covarianza_q_inversa = np.linalg.inv(covarianza_q_regularizada)

    diferencia_medias = media_q - media_p
    termino_traza = np.trace(covarianza_q_inversa @ covarianza_p)
    termino_cuadratico = diferencia_medias @ covarianza_q_inversa @ diferencia_medias
    _, log_det_p = np.linalg.slogdet(covarianza_p_regularizada)
    _, log_det_q = np.linalg.slogdet(covarianza_q_regularizada)

    return 0.5 * (termino_traza + termino_cuadratico - dimensiones + (log_det_q - log_det_p))

def compatibilidad_kl(pieza_a, pieza_b, relacion, profundidad=6):
    """
    Modela la zona cercana al borde de cada pieza como una VARIABLE ALEATORIA:
    junta todos los parches de 3x3 pixeles (sobre LUMINANCIA) de esa zona como
    muestras, ajusta una gaussiana multivariada de 9 dimensiones (media +
    covarianza) a las muestras de A y otra a las de B, y mide la diferencia
    entre esas dos distribuciones con la divergencia de Kullback-Leibler.

    Por que esto es distinto de compatibilidad_media_varianza: ese compara
    media y varianza POR SEPARADO (estadisticas marginales, 1 numero por
    pixel). Esto compara la COVARIANZA COMPLETA de 9 dimensiones, es decir la
    estructura CONJUNTA entre pixeles vecinos dentro de cada parche (que tan
    correlacionados estan un pixel y el de al lado, en diagonal, etc.) — una
    firma mucho mas rica del "tipo" de textura que un mero par (media, var).

    Como la KL no es simetrica, se promedian KL(A||B) y KL(B||A).

    La gaussiana (media, covarianza) de cada (pieza, lado) se CACHEA (ver
    _ajustar_gaussiana_borde): una pieza no cambia entre una comparacion y la
    siguiente, y construir_matrices_afinidad la compara contra las N-1 piezas
    restantes, asi que ajustarla una sola vez por lado (en vez de una vez por
    cada comparacion) evita trabajo redundante real.

    Parámetros:
    pieza_a: np.ndarray
        Pieza base (origen).
    pieza_b: np.ndarray
        Pieza vecina propuesta.
    relacion: str
        'horizontal' (B a la derecha de A) o 'vertical' (B abajo de A).
    profundidad: int, opcional
        Cuantas lineas cerca del borde se usan para juntar parches de 3x3.
        Se probaron parches de 3x3, 5x5 y 7x7, con y sin color (parches RGB
        de 27 dimensiones en vez de 9 de luminancia), barriendo profundidad de
        4 a 24: 3x3 sobre LUMINANCIA con profundidad chica (4 a 6) fue lo mejor
        en todos los casos. El color no ayudo (necesitaba mucha mas profundidad
        para no perder informacion, y aun asi resolvia menos casos) y los
        parches mas grandes empeoraron (menos muestras disponibles para
        estimar una covarianza cada vez mas grande). Una zona muy ancha
        tambien empeora: empieza a incluir contenido cada vez menos
        relacionado con el borde en si.

    Retorna:
    float
        KL(A||B) + KL(B||A) entre las gaussianas ajustadas a los parches de
        cada pieza.

    Ejemplo de uso
    --------------
    >>> rng = np.random.default_rng(0)
    >>> pieza_1 = rng.random((30, 30, 3))
    >>> pieza_2 = rng.random((30, 30, 3))  # misma "textura" (ruido uniforme), otra muestra
    >>> costo = compatibilidad_kl(pieza_1, pieza_2, relacion="horizontal")
    >>> print(f"Costo de poner pieza_2 a la derecha de pieza_1: {costo:.1f}")
    Costo de poner pieza_2 a la derecha de pieza_1: 0.6
    """
    if relacion not in BORDES_ENFRENTADOS:
        raise ValueError(
            f"Relación inválida: '{relacion}'. Se espera 'horizontal' o 'vertical'."
        )

    lado_a, lado_b = BORDES_ENFRENTADOS[relacion]

    media_a, covarianza_a = _ajustar_gaussiana_borde(pieza_a, lado_a, profundidad)
    media_b, covarianza_b = _ajustar_gaussiana_borde(pieza_b, lado_b, profundidad)

    kl_a_hacia_b = _kl_gaussianas_multivariadas(media_a, covarianza_a, media_b, covarianza_b)
    kl_b_hacia_a = _kl_gaussianas_multivariadas(media_b, covarianza_b, media_a, covarianza_a)

    return float(kl_a_hacia_b + kl_b_hacia_a)


#Constantes de normalizacion para compatibilidad_baseline_kl: cada termino se
#divide por su propio costo medio entre vecinos VERDADEROS (medido sobre la
#imagen de referencia mercado_especias_3597x2501.jpg, grilla 8x8), para que
#ninguno de los dos domine por pura diferencia de escala antes de pesarlos.
#Un barrido de pesos (ver conversacion) encontro que 70% baseline / 30% KL es
#lo mejor: agregar gradiante a la mezcla nunca mejoro el resultado (sus errores
#estan correlacionados con los de baseline, no aportan nada nuevo), por eso NO
#esta incluido aca.
_ESCALA_BASELINE_EN_MEZCLA = 0.7 / 3.3798488218816938
_ESCALA_KL_EN_MEZCLA = 0.3 / 1.8251348251093071


def compatibilidad_baseline_kl(pieza_a, pieza_b, relacion):
    """
    Combina compatibilidad_baseline (diferencia pixel a pixel en la costura)
    con compatibilidad_kl (diferencia de distribucion, via KL, de los parches
    de 3x3 cerca del borde): 70% baseline + 30% KL, normalizando antes cada
    termino por su propia escala tipica (ver _ESCALA_BASELINE_EN_MEZCLA y
    _ESCALA_KL_EN_MEZCLA) para que la mezcla no quede dominada por el que
    tenga numeros mas grandes.

    Por que esta combinacion y no otra: en las pruebas, baseline solo ya
    separa casi perfectamente los vecinos verdaderos de los falsos, salvo un
    puñado de pares que son recortes de la MISMA textura repetitiva/aleatoria
    (p. ej. hebras de azafran/chile): ahi baseline y gradiante fallan de la
    misma manera (estan mirando esencialmente la misma informacion), pero KL
    -que compara la estructura CONJUNTA de una zona mas ancha, no solo la
    linea de contacto- resuelve una fraccion real de esos casos que ningun
    otro termino resuelve. La mezcla 70/30 fue la que mejor AUC dio y la que
    mas de esos casos difíciles resolvio, en un barrido de pesos.

    Parámetros:
    pieza_a: np.ndarray
        Pieza base (origen).
    pieza_b: np.ndarray
        Pieza vecina propuesta.
    relacion: str
        'horizontal' (B a la derecha de A) o 'vertical' (B abajo de A).

    Retorna:
    float
        0.7 * (baseline normalizado) + 0.3 * (KL normalizado).

    Ejemplo de uso
    --------------
    >>> # Piezas de textura pareja (no planas): con piezas perfectamente planas,
    >>> # compatibilidad_kl da un numero enorme (covarianza casi nula, ver su propio
    >>> # docstring), y ese termino domina la mezcla igual que domina a compatibilidad_kl sola.
    >>> rng = np.random.default_rng(0)
    >>> pieza_1 = rng.random((30, 30, 3))
    >>> pieza_2 = rng.random((30, 30, 3))
    >>> costo = compatibilidad_baseline_kl(pieza_1, pieza_2, relacion="horizontal")
    >>> print(f"Costo de poner pieza_2 a la derecha de pieza_1: {costo:.1f}")
    Costo de poner pieza_2 a la derecha de pieza_1: 3.0
    """
    if relacion not in BORDES_ENFRENTADOS:
        raise ValueError(
            f"Relación inválida: '{relacion}'. Se espera 'horizontal' o 'vertical'."
        )

    baseline = compatibilidad_baseline(pieza_a, pieza_b, relacion)
    kl = compatibilidad_kl(pieza_a, pieza_b, relacion)

    return _ESCALA_BASELINE_EN_MEZCLA * baseline + _ESCALA_KL_EN_MEZCLA * kl


def construir_matrices_afinidad(
    piezas: List[np.ndarray],
    funcion_compatibilidad: Callable[[np.ndarray, np.ndarray, str], float] = compatibilidad_baseline_kl,
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
    limpiar_cache_kl()

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
