"""
Módulo de Degradaciones y Ruidos Sintéticos.
Incluye ruidos espaciales (Nivel 1), distorsiones fotométricas por pieza (Nivel 2)
y ruido periódico en el dominio de Fourier (Nivel 3).
Todas las funciones operan en imágenes RGB float64 [0.0, 1.0].
"""

import numpy as np
from skimage import color

__all__ = [
    "agregar_ruido_gaussiano",
    "agregar_ruido_uniforme",
    "agregar_ruido_rayleigh",
    "agregar_ruido_exponencial",
    "agregar_ruido_sal_y_pimienta",
    "componer_degradaciones",
    "rotar_matiz",
    "alterar_valor",
    "VARIANTES_CROMATICAS",
    "DegradacionCromaticaPorPieza",
    "agregar_onda",
    "agregar_ondas",
    "agregar_producto_de_ondas",
    "TIPOS_DE_TRAMA",
    "TramaMixtaPorPieza",
]


def _acotar_rango(imagen: np.ndarray) -> np.ndarray:
    """Acota los valores de la imagen al rango válido [0.0, 1.0]."""
    return np.clip(imagen, 0.0, 1.0)


# ==============================================================================
# NIVEL 1: Ruidos Espaciales Aditivos e Impulsivos
# ==============================================================================

def agregar_ruido_gaussiano(
    imagen: np.ndarray,
    generador: np.random.Generator,
    desviacion_estandar: float = 0.08,
    media: float = 0.0,
) -> np.ndarray:
    """Ruido gaussiano aditivo canal por canal."""
    ruido = generador.normal(loc=media, scale=desviacion_estandar, size=imagen.shape)
    return _acotar_rango(imagen + ruido)


def agregar_ruido_uniforme(
    imagen: np.ndarray,
    generador: np.random.Generator,
    limite_inferior: float = -0.1,
    limite_superior: float = 0.1,
) -> np.ndarray:
    """Ruido uniforme aditivo."""
    ruido = generador.uniform(low=limite_inferior, high=limite_superior, size=imagen.shape)
    return _acotar_rango(imagen + ruido)


def agregar_ruido_rayleigh(
    imagen: np.ndarray,
    generador: np.random.Generator,
    desplazamiento: float = 0.0,
    parametro_b: float = 0.015,
    centrar_media: bool = True,
) -> np.ndarray:
    """Ruido Rayleigh aditivo (asimétrico, cola a la derecha)."""
    escala = np.sqrt(parametro_b / 2.0)
    ruido = desplazamiento + generador.rayleigh(scale=escala, size=imagen.shape)
    if centrar_media:
        media_teorica = desplazamiento + np.sqrt(np.pi * parametro_b / 4.0)
        ruido -= media_teorica
    return _acotar_rango(imagen + ruido)


def agregar_ruido_exponencial(
    imagen: np.ndarray,
    generador: np.random.Generator,
    tasa_a: float = 15.0,
    centrar_media: bool = True,
) -> np.ndarray:
    """Ruido exponencial aditivo."""
    escala = 1.0 / tasa_a
    ruido = generador.exponential(scale=escala, size=imagen.shape)
    if centrar_media:
        ruido -= escala
    return _acotar_rango(imagen + ruido)


def agregar_ruido_sal_y_pimienta(
    imagen: np.ndarray,
    generador: np.random.Generator,
    probabilidad_sal: float = 0.02,
    probabilidad_pimienta: float = 0.02,
) -> np.ndarray:
    """Ruido impulsivo sal y pimienta que afecta a los 3 canales simultáneamente."""
    imagen_ruidosa = imagen.copy()
    matriz_sorteo = generador.random(imagen.shape[:2])

    mascara_sal = matriz_sorteo < probabilidad_sal
    mascara_pimienta = (matriz_sorteo >= probabilidad_sal) & (
        matriz_sorteo < (probabilidad_sal + probabilidad_pimienta)
    )

    imagen_ruidosa[mascara_sal, :] = 1.0
    imagen_ruidosa[mascara_pimienta, :] = 0.0
    return imagen_ruidosa


#----------------
# Nivel 2
#-----------------
# ==============================================================================
# NIVEL 2: Variaciones fotométricas por pieza
# ==============================================================================

VARIANTES_CROMATICAS = ("matiz", "valor")


def rotar_matiz(pieza, desplazamiento):
    """
    Rota el matiz (canal H de HSV) dejando saturacion y valor intactos.

    `desplazamiento` va de 0 a 1 y es circular: 0.5 lleva cada color a su
    complementario.

    En RGB el efecto se ve como un cambio de color imposible de corregir canal
    por canal, porque la rotacion mezcla los tres.
    """
    hsv = color.rgb2hsv(np.clip(pieza, 0.0, 1.0))
    hsv[:, :, 0] = (hsv[:, :, 0] + desplazamiento) % 1.0
    return np.clip(color.hsv2rgb(hsv), 0.0, 1.0)


def alterar_valor(pieza, gamma=1.0, ganancia=1.0):
    """
    Modifica el canal V de HSV con gamma y ganancia, dejando matiz y saturacion
    intactos.

        V' = ganancia * V^(1/gamma)

    Es un cambio de claridad que conserva el color: la pieza se ve mas clara o
    mas oscura pero del mismo tono.
    """
    hsv = color.rgb2hsv(np.clip(pieza, 0.0, 1.0))
    valor = np.power(np.clip(hsv[:, :, 2], 0.0, 1.0), 1.0 / max(gamma, 0.1))
    hsv[:, :, 2] = np.clip(valor * ganancia, 0.0, 1.0)
    return np.clip(color.hsv2rgb(hsv), 0.0, 1.0)


class DegradacionCromaticaPorPieza:
    """
    Aplica a cada pieza una transformacion de color distinta.
    Se usa como `degradacion_por_pieza`.

    Dos variantes, con soluciones OPUESTAS:

        matiz   rota H por pieza. Sobreviven S y V.
                Solucion: comparar en V (o en S y V), descartando H.
                Diagnostico: las piezas cambian de COLOR pero mantienen su
                claridad y su nivel de saturacion.

        valor   gamma y ganancia sobre V. Sobreviven H y S.
                Solucion: comparar en H y S, descartando V.
                Diagnostico: las piezas cambian de CLARIDAD pero mantienen su
                tono.

    OJO CON EL MATIZ: es circular. Comparar H directamente da mal, porque 0.99 y
    0.01 son casi el mismo color y difieren en 0.98. Hay que llevarlo al plano:
    (cos(2*pi*H) * S, sin(2*pi*H) * S).

    Registra en `parametros_por_pieza` que le toco a cada una.

    Los parametros se sortean sin correlacion con la posicion de la pieza: si
    piezas vecinas compartieran transformacion, comparar colores agruparia
    piezas por su degradacion en vez de por su contenido.
    """

    def __init__(self, variante="matiz",
                 rango_desplazamiento=(0.0, 1.0),
                 rango_gamma=(0.50, 2.00),
                 rango_ganancia=(0.60, 1.40)):
        if variante not in VARIANTES_CROMATICAS:
            raise ValueError(f"variante invalida: {variante!r}. "
                             f"Se espera una de {VARIANTES_CROMATICAS}")
        self.variante = variante
        self.rango_desplazamiento = rango_desplazamiento
        self.rango_gamma = rango_gamma
        self.rango_ganancia = rango_ganancia
        self.parametros_por_pieza = {}

    def __call__(self, pieza, indice, generador):
        if self.variante == "matiz":
            desplazamiento = float(generador.uniform(*self.rango_desplazamiento))
            self.parametros_por_pieza[int(indice)] = {
                "variante": "matiz",
                "desplazamiento_h": round(desplazamiento, 3),
            }
            return rotar_matiz(pieza, desplazamiento)

        gamma = float(generador.uniform(*self.rango_gamma))
        ganancia = float(generador.uniform(*self.rango_ganancia))
        self.parametros_por_pieza[int(indice)] = {
            "variante": "valor",
            "gamma": round(gamma, 3),
            "ganancia": round(ganancia, 3),
        }
        return alterar_valor(pieza, gamma, ganancia)

    def reiniciar(self):
        self.parametros_por_pieza = {}

    def resumen(self):
        """Rango efectivo de los parametros sorteados, para verificar el sorteo."""
        if not self.parametros_por_pieza:
            return {}

        if self.variante == "matiz":
            valores = [r["desplazamiento_h"] for r in self.parametros_por_pieza.values()]
            return {"variante": "matiz", "desplazamiento_h": (min(valores), max(valores))}

        gammas = [r["gamma"] for r in self.parametros_por_pieza.values()]
        ganancias = [r["ganancia"] for r in self.parametros_por_pieza.values()]
        return {"variante": "valor",
                "gamma": (min(gammas), max(gammas)),
                "ganancia": (min(ganancias), max(ganancias))}


# =========================================================================
# RUIDO PERIODICO (Gonzalez 5.2.3) — nivel 3
# =========================================================================
#
# Todas las tramas de este bloque estan disenadas para que un mismo notch las
# saque bien. Tres condiciones, y las tres importan:
#
# 1. PICOS EN BINS EXACTOS. La onda se parametriza por su VECTOR DE FRECUENCIA
#    (du, dv) en bins, no por periodo y orientacion. Asi el pico cae exacto en
#    (du, dv) sea cual sea la direccion.
#
#    Con periodo + orientacion, una onda diagonal de periodo 8 sobre 128 px pone
#    su pico a 128/(8*raiz(2)) = 11.3 bins: no es entero, hay leakage, el pico se
#    ensancha y la recuperacion cae de +27 dB a +7 dB. Medido.
#
# 2. FRECUENCIAS ALTAS. Una imagen natural concentra casi toda su energia cerca
#    del DC. Un pico lejos del centro cae en una zona vacia del espectro: se
#    detecta facil y el notch no se lleva nada de la imagen.
#
#        pico a 32 bins -> recuperacion 38.7 dB
#        pico a 16 bins -> recuperacion 38.8 dB
#        pico a  8 bins -> recuperacion 35.1 dB
#        pico a  4 bins -> recuperacion 29.5 dB   <- ya toca el contenido
#
# 3. VARIOS PICOS POR PIEZA. Cuantos mas picos, mas se degrada el emparejamiento
#    de bordes sin filtrar, y sin embargo el notch los saca todos igual de bien.
#    Es la combinacion que se busca: imposible de armar sucio, casi perfecto
#    filtrado.
#
# Lo que quedo AFUERA y por que: la onda cuadrada (armonicos en 3f, 5f, 7f,
# demasiado debiles para detectar, dejan residuo) y la trama de semitono (red de
# picos girada, muchos y debiles, el detector o se pasa o se queda corto).


#Frecuencias disponibles, en bins desde el centro del espectro. Todas altas.
FRECUENCIAS_DISPONIBLES = (8, 12, 16, 20, 24, 28)


def agregar_onda(imagen, desplazamiento_fila, desplazamiento_columna,
                 amplitud=0.09, fase=0.0):
    """
    Suma una onda sinusoidal cuyo pico cae EXACTO en (du, dv) bins del centro
    del espectro centrado.

        onda(y, x) = A * sin(2*pi * (du*y/alto + dv*x/ancho) + fase)

    Parametrizar por el vector de frecuencia en lugar de por periodo y angulo
    garantiza que el pico caiga en un bin entero en cualquier direccion, y de
    paso son las mismas coordenadas que el alumno lee del espectro.

    La frecuencia se traduce a periodo espacial asi: una onda con (0, dv) tiene
    periodo ancho/dv en horizontal. Sobre una pieza de 128 px, dv=16 es periodo 8.
    """
    alto, ancho = imagen.shape[:2]

    fase_espacial = 2.0 * np.pi * (
        desplazamiento_fila * np.arange(alto)[:, None] / alto +
        desplazamiento_columna * np.arange(ancho)[None, :] / ancho)

    onda = amplitud * np.sin(fase_espacial + fase)
    if imagen.ndim == 3:
        onda = onda[:, :, None]
    return _acotar_rango(imagen + onda)


def agregar_ondas(imagen, componentes):
    """
    Suma varias ondas. `componentes` es una lista de (du, dv, amplitud, fase).

    La suma es lineal, asi que en el espectro aparece un par de picos por cada
    componente, sin terminos cruzados.
    """
    resultado = imagen
    for desplazamiento_fila, desplazamiento_columna, amplitud, fase in componentes:
        resultado = agregar_onda(resultado, desplazamiento_fila,
                                 desplazamiento_columna, amplitud, fase)
    return resultado


def agregar_producto_de_ondas(imagen, primera, segunda, amplitud=0.18):
    """
    Suma el PRODUCTO de dos ondas. `primera` y `segunda` son (du, dv).

    Se distingue de la suma en el espectro, y esa es la gracia. Por la identidad
    sin(a)sin(b) = [cos(a-b) - cos(a+b)]/2, el producto NO tiene picos en las
    frecuencias originales: los tiene en la SUMA y la DIFERENCIA de los dos
    vectores. Quien aplique el notch en las frecuencias "obvias" no elimina nada.
    """
    alto, ancho = imagen.shape[:2]
    filas = np.arange(alto)[:, None]
    columnas = np.arange(ancho)[None, :]

    def _fase(desplazamiento):
        return 2.0 * np.pi * (desplazamiento[0] * filas / alto +
                              desplazamiento[1] * columnas / ancho)

    onda = amplitud * np.sin(_fase(primera)) * np.sin(_fase(segunda))
    if imagen.ndim == 3:
        onda = onda[:, :, None]
    return _acotar_rango(imagen + onda)


# --------------------------------------------------------------------------
# Catalogo de tramas
# --------------------------------------------------------------------------
#
# Cada entrada devuelve la lista de componentes (du, dv, amplitud, fase), o bien
# marca que es un producto. Todas dan entre 2 y 4 pares de picos, todos altos y
# aislados.

TIPOS_DE_TRAMA = ("ortogonales", "rejilla", "diagonales", "oblicuas",
                  "triple", "doble_frecuencia", "cuadruple")


def _sortear_frecuencia(generador, frecuencias):
    return int(generador.choice(frecuencias))


def _componentes_de_trama(tipo, generador, frecuencias, amplitud):
    """
    Devuelve (componentes, es_producto, descripcion) para el tipo pedido.

    Cuando hay varias componentes la amplitud se reparte, para que la energia
    total del ruido sea comparable entre tipos: si no, una trama de cuatro ondas
    seria el doble de fuerte que una de dos y los tipos no serian equivalentes.
    """
    sortear = lambda: _sortear_frecuencia(generador, frecuencias)

    #Las componentes diagonales usan un piso de 8 bins por eje. Un pico en (d, d)
    #esta a d*raiz(2) del centro, asi que d=4 lo dejaria a solo 5.7 bins: dentro
    #de la zona donde vive el contenido de la imagen, que es justo lo que se
    #quiere evitar.
    diagonal = lambda f: max(8, f // 2 * 2)
    fase = lambda: float(generador.uniform(0, 2 * np.pi))
    signo = lambda: int(generador.choice([-1, 1]))

    if tipo == "ortogonales":
        f1, f2 = sortear(), sortear()
        return [(0, f1, amplitud, fase()), (f2, 0, amplitud, fase())], False, \
               {"picos": [(0, f1), (f2, 0)]}

    if tipo == "rejilla":
        f1, f2 = sortear(), sortear()
        #Producto: los picos NO estan en (0,f1) ni (f2,0), sino en (f2, +-f1)
        return [(0, f1), (f2, 0)], True, \
               {"picos": [(f2, f1), (f2, -f1)]}

    if tipo == "diagonales":
        f = diagonal(sortear())
        return [(f, f, amplitud, fase()), (f, -f, amplitud, fase())], False, \
               {"picos": [(f, f), (f, -f)]}

    if tipo == "oblicuas":
        f1, f2 = sortear(), sortear()
        oblicuo = max(8, f1 // 2)
        return [(oblicuo, f2, amplitud, fase()),
                (f2, signo() * oblicuo, amplitud, fase())], False, \
               {"picos": [(oblicuo, f2), (f2, oblicuo)]}

    if tipo == "triple":
        f1, f2, f3 = sortear(), sortear(), sortear()
        parcial = amplitud * 0.8
        d = diagonal(f3)
        return [(0, f1, parcial, fase()), (f2, 0, parcial, fase()),
                (d, d, parcial, fase())], False, \
               {"picos": [(0, f1), (f2, 0), (d, d)]}

    if tipo == "doble_frecuencia":
        #Dos ondas en la MISMA direccion, distinta frecuencia: dos pares de picos
        #alineados sobre el mismo eje
        #Dos frecuencias distintas del catalogo, la mas baja y la mas alta de
        #tres sorteos, para que los dos pares de picos queden bien separados
        candidatas = sorted({sortear(), sortear(), sortear()})
        f1, f2 = candidatas[0], candidatas[-1]
        if f1 == f2:
            f1, f2 = min(frecuencias), max(frecuencias)
        direccion = str(generador.choice(["horizontal", "vertical"]))
        if direccion == "horizontal":
            componentes = [(0, f1, amplitud, fase()), (0, f2, amplitud, fase())]
            picos = [(0, f1), (0, f2)]
        else:
            componentes = [(f1, 0, amplitud, fase()), (f2, 0, amplitud, fase())]
            picos = [(f1, 0), (f2, 0)]
        return componentes, False, {"picos": picos}

    if tipo == "cuadruple":
        f1, f2 = sortear(), sortear()
        d = diagonal(min(f1, f2))
        parcial = amplitud * 0.65
        return [(0, f1, parcial, fase()), (f2, 0, parcial, fase()),
                (d, d, parcial, fase()), (d, -d, parcial, fase())], False, \
               {"picos": [(0, f1), (f2, 0), (d, d), (d, -d)]}

    raise ValueError(f"tipo de trama invalido: {tipo!r}")


class TramaMixtaPorPieza:
    """
    Aplica a cada pieza una trama periodica de tipo y parametros sorteados.
    Se usa como `degradacion_por_pieza`.

    Registra en `parametros_por_pieza` que le toco a cada pieza, incluidas las
    coordenadas exactas de sus picos. Eso permite verificar que el alumno los
    detecto bien, y no solo que subio el PSNR.

    Los parametros se sortean SIN correlacion con la posicion de la pieza. Si
    piezas vecinas compartieran trama, la trama seria una firma: comparar
    espectros agruparia piezas por su ruido en vez de por su contenido, y el
    acierto subiria sin filtrar nada.

    Amplitud por defecto 0.12 a 0.18: con eso el rompecabezas es imposible de
    armar sin filtrar (vecindad ~15%) y queda casi perfecto con el notch
    (vecindad >95%, PSNR ~45 dB).
    """

    def __init__(self, tipos=TIPOS_DE_TRAMA, frecuencias=FRECUENCIAS_DISPONIBLES,
                 amplitud_minima=0.12, amplitud_maxima=0.18):
        self.tipos = tuple(tipos)
        self.frecuencias = tuple(frecuencias)
        self.amplitud_minima = amplitud_minima
        self.amplitud_maxima = amplitud_maxima
        self.parametros_por_pieza = {}

    def __call__(self, pieza, indice, generador):
        tipo = str(generador.choice(self.tipos))
        amplitud = float(generador.uniform(self.amplitud_minima, self.amplitud_maxima))

        componentes, es_producto, info = _componentes_de_trama(
            tipo, generador, self.frecuencias, amplitud)

        if es_producto:
            resultado = agregar_producto_de_ondas(pieza, componentes[0],
                                                  componentes[1], amplitud * 2.0)
        else:
            resultado = agregar_ondas(pieza, componentes)

        self.parametros_por_pieza[int(indice)] = {
            "tipo": tipo,
            "amplitud": round(amplitud, 4),
            "picos": info["picos"],
        }
        return resultado

    def reiniciar(self):
        self.parametros_por_pieza = {}

    def resumen(self):
        """Cuantas piezas recibieron cada tipo de trama."""
        conteo = {}
        for registro in self.parametros_por_pieza.values():
            conteo[registro["tipo"]] = conteo.get(registro["tipo"], 0) + 1
        return conteo


# ==============================================================================
# Composición de degradaciones
# ==============================================================================

def componer_degradaciones(*degradaciones):
    """Compone múltiples funciones de degradación en secuencia: f_n(...(f_1(img)))."""
    def degradacion_compuesta(imagen, generador):
        res = imagen
        for deg in degradaciones:
            res = deg(res, generador)
        return res
    return degradacion_compuesta
