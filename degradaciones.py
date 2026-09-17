import numpy as np
from skimage import color

__all__ = [
    "agregar_ruido_gaussiano",
    "agregar_ruido_uniforme",
    "agregar_ruido_rayleigh",
    "agregar_ruido_exponencial",
    "agregar_ruido_sal_y_pimienta",
    "componer_degradaciones",
    "alterar_luminancia", 
    "alterar_crominancia", 
    "VARIANTES_NIVEL_2", 
    "DegradacionPorPiezaNivel2", 
    "sortear_variante_nivel2",
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



VARIANTES_NIVEL_2 = ("luminancia", "crominancia", "xor")
 
 
def _a_ycbcr(pieza):
    from skimage import color
    return color.rgb2ycbcr(np.clip(pieza, 0.0, 1.0))
 
 
def _a_rgb(ycbcr):
    from skimage import color
    return np.clip(color.ycbcr2rgb(ycbcr), 0.0, 1.0)
 
 
def alterar_luminancia(pieza, gamma=1.0, contraste=1.0, brillo=0.0):
    ycbcr = _a_ycbcr(pieza)
    luminancia = (ycbcr[:, :, 0] - 16.0) / 219.0
    luminancia = np.power(np.clip(luminancia, 0.0, 1.0), 1.0 / max(gamma, 0.1))
    luminancia = contraste * (luminancia - 0.5) + 0.5 + brillo
    ycbcr[:, :, 0] = np.clip(luminancia * 219.0 + 16.0, 16.0, 235.0)
    return _a_rgb(ycbcr)
 
 
def alterar_crominancia(pieza, ganancia_cb=1.0, ganancia_cr=1.0,
                        desplazamiento_cb=0.0, desplazamiento_cr=0.0):
    ycbcr = _a_ycbcr(pieza)
    ycbcr[:, :, 1] = np.clip((ycbcr[:, :, 1] - 128.0) * ganancia_cb + 128.0 + desplazamiento_cb, 16.0, 240.0)
    ycbcr[:, :, 2] = np.clip((ycbcr[:, :, 2] - 128.0) * ganancia_cr + 128.0 + desplazamiento_cr, 16.0, 240.0)
    return _a_rgb(ycbcr)
 
 
def sortear_variante_nivel2(semilla):
    """
    Sortea la variante a partir de la semilla del rompecabezas, con un generador
    propio para no correr el stream aleatorio del resto de los niveles.
    """
    generador = np.random.default_rng([int(semilla), 20252])
    return str(generador.choice(list(VARIANTES_NIVEL_2)))
 
 
class DegradacionPorPiezaNivel2:
    """
    Se usa como `degradacion_por_pieza`: se la llama (pieza, indice, generador).
 
    Los parametros se sortean sin correlacion con la posicion de la pieza: si
    piezas vecinas compartieran transformacion, comparar brillos agruparia
    piezas por su degradacion en vez de por su contenido.
    """
 
    def __init__(self, variante="xor", cantidad_piezas=None,
                 rango_gamma=(0.55, 1.80),
                 rango_contraste=(0.60, 1.45),
                 rango_brillo=(-0.12, 0.12),
                 rango_ganancia=(0.30, 2.00),
                 rango_desplazamiento=(-45.0, 45.0)):
        if variante not in VARIANTES_NIVEL_2:
            raise ValueError(f"variante invalida: {variante!r}. "
                             f"Se espera una de {VARIANTES_NIVEL_2}")
        self.variante = variante
        self.cantidad_piezas = cantidad_piezas
        self.rango_gamma = rango_gamma
        self.rango_contraste = rango_contraste
        self.rango_brillo = rango_brillo
        self.rango_ganancia = rango_ganancia
        self.rango_desplazamiento = rango_desplazamiento
 
        self.parametros_por_pieza = {}
        self._asignacion = None
 
 
    def _que_le_toca(self, indice, generador):
        if self.variante != "xor":
            return self.variante
 
    
        if self.cantidad_piezas:
            if self._asignacion is None:
                mitad = self.cantidad_piezas // 2
                etiquetas = (["luminancia"] * mitad +
                             ["crominancia"] * (self.cantidad_piezas - mitad))
                generador.shuffle(etiquetas)
                self._asignacion = etiquetas
            return self._asignacion[int(indice) % len(self._asignacion)]
 
        return "luminancia" if generador.random() < 0.5 else "crominancia"
 
 
    def __call__(self, pieza, indice, generador):
        le_toca = self._que_le_toca(indice, generador)
 
        if le_toca == "luminancia":
            parametros = {
                "gamma": float(generador.uniform(*self.rango_gamma)),
                "contraste": float(generador.uniform(*self.rango_contraste)),
                "brillo": float(generador.uniform(*self.rango_brillo)),
            }
            resultado = alterar_luminancia(pieza, **parametros)
        else:
            parametros = {
                "ganancia_cb": float(generador.uniform(*self.rango_ganancia)),
                "ganancia_cr": float(generador.uniform(*self.rango_ganancia)),
                "desplazamiento_cb": float(generador.uniform(*self.rango_desplazamiento)),
                "desplazamiento_cr": float(generador.uniform(*self.rango_desplazamiento)),
            }
            resultado = alterar_crominancia(pieza, **parametros)
 
        registro = {"altera": le_toca}
        registro.update({k: round(v, 3) for k, v in parametros.items()})
        self.parametros_por_pieza[int(indice)] = registro
        return resultado
 
    def reiniciar(self):
        self.parametros_por_pieza = {}
        self._asignacion = None

FRECUENCIAS_DISPONIBLES = (24,  40,  56, 100)


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



TIPOS_DE_TRAMA = ("ortogonales", "diagonales", "oblicuas",
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
        return [(0, f1, amplitud, fase()), (f2, 0, amplitud, fase())], \
               {"picos": [(0, f1), (f2, 0)]}


    if tipo == "diagonales":
        f = diagonal(sortear())
        return [(f, f, amplitud, fase()), (f, -f, amplitud, fase())], \
               {"picos": [(f, f), (f, -f)]}

    if tipo == "oblicuas":
        f1, f2 = sortear(), sortear()
        oblicuo = max(8, f1 // 2)
        return [(oblicuo, f2, amplitud, fase()),
                (f2, signo() * oblicuo, amplitud, fase())], \
               {"picos": [(oblicuo, f2), (f2, oblicuo)]}

    if tipo == "triple":
        f1, f2, f3 = sortear(), sortear(), sortear()
        parcial = amplitud * 0.8
        d = diagonal(f3)
        return [(0, f1, parcial, fase()), (f2, 0, parcial, fase()),
                (d, d, parcial, fase())], \
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
        return componentes, {"picos": picos}

    if tipo == "cuadruple":
        f1, f2 = sortear(), sortear()
        d = diagonal(min(f1, f2))
        parcial = amplitud * 0.65
        return [(0, f1, parcial, fase()), (f2, 0, parcial, fase()),
                (d, d, parcial, fase()), (d, -d, parcial, fase())], \
               {"picos": [(0, f1), (f2, 0), (d, d), (d, -d)]}

    raise ValueError(f"tipo de trama invalido: {tipo!r}")


class TramaMixtaPorPieza:
    # Mantenemos amplitudes altas para destruir visualmente la imagen
    def __init__(self, tipos=TIPOS_DE_TRAMA, frecuencias=FRECUENCIAS_DISPONIBLES,
                 amplitud_minima=0.25, amplitud_maxima=0.35):
        self.tipos = tuple(tipos)
        self.frecuencias = tuple(frecuencias)
        self.amplitud_minima = amplitud_minima
        self.amplitud_maxima = amplitud_maxima
        self.parametros_por_pieza = {}

    def __call__(self, pieza, indice, generador):
        tipo = str(generador.choice(self.tipos))
        amplitud = float(generador.uniform(self.amplitud_minima, self.amplitud_maxima))

        # Ya no recibimos "es_producto" porque todas son sumas puras
        componentes, info = _componentes_de_trama(
            tipo, generador, self.frecuencias, amplitud)

        # Usamos siempre agregar_ondas (suma pura)
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
