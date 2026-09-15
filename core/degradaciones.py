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


def agregar_ruido_gaussiano(
    imagen: np.ndarray,
    generador: np.random.Generator,
    desviacion_estandar: float = 0.08,
    media: float = 0.0,
) -> np.ndarray:
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




VARIANTES_CROMATICAS = ("matiz", "valor")


def rotar_matiz(pieza, desplazamiento):
    hsv = color.rgb2hsv(np.clip(pieza, 0.0, 1.0))
    hsv[:, :, 0] = (hsv[:, :, 0] + desplazamiento) % 1.0
    return np.clip(color.hsv2rgb(hsv), 0.0, 1.0)


def alterar_valor(pieza, gamma=1.0, ganancia=1.0):
    hsv = color.rgb2hsv(np.clip(pieza, 0.0, 1.0))
    valor = np.power(np.clip(hsv[:, :, 2], 0.0, 1.0), 1.0 / max(gamma, 0.1))
    hsv[:, :, 2] = np.clip(valor * ganancia, 0.0, 1.0)
    return np.clip(color.hsv2rgb(hsv), 0.0, 1.0)


class DegradacionCromaticaPorPieza:
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

FRECUENCIAS_DISPONIBLES = (8, 12, 16, 20, 24, 28)


def agregar_onda(imagen, desplazamiento_fila, desplazamiento_columna,
                 amplitud=0.09, fase=0.0):
    
    alto, ancho = imagen.shape[:2]

    fase_espacial = 2.0 * np.pi * (
        desplazamiento_fila * np.arange(alto)[:, None] / alto +
        desplazamiento_columna * np.arange(ancho)[None, :] / ancho)

    onda = amplitud * np.sin(fase_espacial + fase)
    if imagen.ndim == 3:
        onda = onda[:, :, None]
    return _acotar_rango(imagen + onda)


def agregar_ondas(imagen, componentes):
   
    resultado = imagen
    for desplazamiento_fila, desplazamiento_columna, amplitud, fase in componentes:
        resultado = agregar_onda(resultado, desplazamiento_fila,
                                 desplazamiento_columna, amplitud, fase)
    return resultado


def agregar_producto_de_ondas(imagen, primera, segunda, amplitud=0.18):
    
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



TIPOS_DE_TRAMA = ("ortogonales", "rejilla", "diagonales", "oblicuas",
                  "triple", "doble_frecuencia", "cuadruple")


def _sortear_frecuencia(generador, frecuencias):
    return int(generador.choice(frecuencias))


def _componentes_de_trama(tipo, generador, frecuencias, amplitud):
   
    sortear = lambda: _sortear_frecuencia(generador, frecuencias)

    
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
      
        conteo = {}
        for registro in self.parametros_por_pieza.values():
            conteo[registro["tipo"]] = conteo.get(registro["tipo"], 0) + 1
        return conteo


def componer_degradaciones(*degradaciones):
    def degradacion_compuesta(imagen, generador):
        res = imagen
        for deg in degradaciones:
            res = deg(res, generador)
        return res
    return degradacion_compuesta
