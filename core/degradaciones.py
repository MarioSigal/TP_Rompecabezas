import numpy as np
from skimage import color
from core.analizador_rotacion import enderezar_pieza
from core.geometria_jigsaw import MINIMO_CONTENIDO_MASCARA_FLOAT, MINIMO_CONTENIDO_MASCARA_INT

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

def _aplicar_invariante_de_mascara(imagen,mascara):

    imagen[~mascara] = 0
    
    if issubclass(imagen.dtype.type, np.floating):
        imagen[mascara] = np.maximum(imagen[mascara], MINIMO_CONTENIDO_MASCARA_FLOAT)
    else:
        imagen[mascara] = np.maximum(imagen[mascara], MINIMO_CONTENIDO_MASCARA_INT)
    
    return imagen

# region Ruidos Nivel 1

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

# endregion


# region Degradacion Nivel 2
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
 
 
    def __call__(self, pieza, indice, generador, mascara=None):
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

        if mascara is not None:
            resultado =_aplicar_invariante_de_mascara(resultado, mascara)

        return resultado, mascara
 
    def reiniciar(self):
        self.parametros_por_pieza = {}
        self._asignacion = None

# endregion

# region Degradacion Nivel 3

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

    def __call__(self, pieza, indice, generador, mascara = None):
        tipo = str(generador.choice(self.tipos))
        amplitud = float(generador.uniform(self.amplitud_minima, self.amplitud_maxima))

        # Ya no recibimos "es_producto" porque todas son sumas puras
        componentes, info = _componentes_de_trama(
            tipo, generador, self.frecuencias, amplitud)

        # Usamos siempre agregar_ondas (suma pura)
        resultado = agregar_ondas(pieza, componentes)

        if mascara is not None:
            resultado =_aplicar_invariante_de_mascara(resultado, mascara)


        self.parametros_por_pieza[int(indice)] = {
            "tipo": tipo,
            "amplitud": round(amplitud, 4),
            "picos": info["picos"],
        }

        return resultado, mascara

    def reiniciar(self):
        self.parametros_por_pieza = {}

    def resumen(self):
        conteo = {}
        for registro in self.parametros_por_pieza.values():
            conteo[registro["tipo"]] = conteo.get(registro["tipo"], 0) + 1
        return conteo

# endregion

# region Degradacion Nivel 5

class DegradacionAgregarFrecuenciaUnicaPorPieza:
    def __init__(self, desplazamiento_fila, desplazamiento_columna, amplitud=10):
        self.desplazamiento_fila = desplazamiento_fila
        self.desplazamiento_columna = desplazamiento_columna
        self.amplitud = amplitud
        self.parametros_por_pieza = {}

    def __call__(self, pieza, indice, generador, mascara = None):
        resultado = agregar_onda(pieza, 
                                 desplazamiento_fila=self.desplazamiento_fila,
                                 desplazamiento_columna=self.desplazamiento_columna,
                                 amplitud=self.amplitud)
                                 
        if mascara is not None:
            resultado =_aplicar_invariante_de_mascara(resultado, mascara)

        self.parametros_por_pieza[int(indice)] = {
                            "desplazamiento_fila": self.desplazamiento_fila,
                            "desplazamiento_columna": self.desplazamiento_columna,
                            "amplitud": self.amplitud,
                        }

        return resultado, mascara

class DegradacionPorPiezaRotacion:
    def __init__(self, rotacion_min=5.0, rotacion_max=60.0, fake_jitter=None):
        self.rotacion_min=rotacion_min
        self.rotacion_max=rotacion_max
        self.fake_jitter = fake_jitter
        self.parametros_por_pieza = {}

    def _get_bounding_box(self, mascara):
            
        y_indices, x_indices = np.where(mascara)
        
        x_inicio = x_indices.min()
        x_fin = x_indices.max()
        y_inicio = y_indices.min()
        y_fin = y_indices.max()

        return x_inicio, x_fin, y_inicio, y_fin

    def _get_cambio_dimensiones_por_giro(self, coordenada_x, coordenada_y, angulo, altura, ancho):

        radio = np.sqrt(altura * altura + ancho * ancho)
        angulo_del_punto = np.atan2(coordenada_y, coordenada_x)

        nuevo_angulo = angulo_del_punto + np.radians(angulo)

        nueva_altura = np.abs(radio * np.sin(nuevo_angulo))  
        nuevo_ancho = np.abs(radio * np.cos(nuevo_angulo))  

        cambio_ancho = nuevo_ancho - ancho  
        cambio_altura = nueva_altura - altura

        return cambio_ancho, cambio_altura

    def _get_cambio_dimension_maximo_por_giro(self, mascara, angulo):
            
        # Find indices where the mask is True
        x_inicio, x_fin, y_inicio, y_fin = self._get_bounding_box(mascara)

        altura = y_fin - y_inicio + 1
        ancho = x_fin - x_inicio + 1

        # hago las coordenadas relativas al centro de la pieza
        x_fin -= x_inicio
        x_inicio = 0  
        x_inicio -= ancho // 2
        x_fin -= ancho // 2

        y_fin -= y_inicio 
        y_fin -= altura // 2 

        cambio_ancho_diagonal_1, cambio_altura_diagonal_1 = self._get_cambio_dimensiones_por_giro(x_inicio, y_fin, angulo, altura, ancho)
        cambio_ancho_diagonal_2, cambio_altura_diagonal_2 = self._get_cambio_dimensiones_por_giro(x_fin, y_fin, angulo, altura, ancho)

        mayor_cambio_ancho = np.maximum(cambio_ancho_diagonal_1, cambio_ancho_diagonal_2)
        mayor_cambio_altura = np.maximum(cambio_altura_diagonal_1, cambio_altura_diagonal_2)

        mayor_cambio = np.maximum(mayor_cambio_ancho, mayor_cambio_altura)
        return np.maximum(0, mayor_cambio)

    def calcular_padding_rotacion(self, mascara, angulo):

        x_inicio, x_fin, y_inicio, y_fin = self._get_bounding_box(mascara)
        altura = y_fin - y_inicio
        ancho = x_fin - x_inicio

        padding_x = (mascara.shape[1] - ancho) // 2
        padding_y = (mascara.shape[0] - altura) // 2
        
        padding_actual = np.minimum(padding_x, padding_y)

        cambio_mayor = self._get_cambio_dimension_maximo_por_giro(mascara, angulo)
        padding_necesario_para_no_cortar_esquinas = cambio_mayor//2 + 1

        # comparamos con el padding actual, si es que hay, no agregos padding al pedo.
        padding_faltante = int(padding_necesario_para_no_cortar_esquinas - padding_actual)
        padding_faltante = np.maximum(0, padding_faltante)

        return padding_faltante

    def __call__(self, pieza, indice, generador, mascara = None):

        if self.fake_jitter == None:
            jitter = float(generador.uniform(5.0, 60.0))
        else:
            jitter = self.fake_jitter

        # Me aseguro que para rotacion se cumpla el invariante de las mascaras
        # aunque no tengo una originalmente, ya que enderar_pieza siempre devuelve una mascara
        # que espera que el invariante sea verdadero
        if mascara is None:
            mascara = np.ones((pieza.shape[0], pieza.shape[1])) == 1
    
        pieza =_aplicar_invariante_de_mascara(pieza, mascara)
        mascara_numerica = (mascara).astype(np.uint8) * 255

        padding_faltante = self.calcular_padding_rotacion(mascara, jitter)

        pieza_rotada, _ = enderezar_pieza(pieza, jitter, padding_faltante)
        mascara_rotada, _ = enderezar_pieza(mascara_numerica, jitter, padding_faltante)

        #Probe con numeros hasta que andara. El 100 es el lucky number
        mascara_rotada = mascara_rotada > 100

        pieza_rotada =_aplicar_invariante_de_mascara(pieza_rotada, mascara_rotada)

        self.parametros_por_pieza[int(indice)] = {
                    "jitter": jitter,
                    "pieza": pieza,
                    "pieza_rotada": pieza_rotada
                }
        
        return pieza_rotada, mascara_rotada

    def reiniciar(self):
        self.parametros_por_pieza = {}

# Composicion de Rotar y agregar lineas para que el angulo de las lineas y rotacion sean el mismo
class DegradacionPorPiezaRotacionYLineasAlMismoAngulo:
    def __init__(self, rotacion_min=5.0, rotacion_max=60.0, fake_jitter=None, amplitud=0.1):
        self.degradador_por_rotacion = DegradacionPorPiezaRotacion(rotacion_min, rotacion_max, fake_jitter)
        self.amplitud = amplitud
        self.parametros_por_pieza = {}

    def __call__(self, pieza, indice, generador, mascara = None):
        pieza_rotada, mascara_rotada = self.degradador_por_rotacion(pieza, indice, generador, mascara)
        angulo_de_rotacion = self.degradador_por_rotacion.parametros_por_pieza[indice]["jitter"]

        periodo = 50

        # Probe combinaciones hasta que funciono, puede ser que sea equivalente a algo
        # donde fila sea seno y columna sea coseno
        # da igual la verdad.
        cambio_en_columna = periodo * np.sin(np.radians(angulo_de_rotacion))
        cambio_en_fila = periodo * np.cos(np.radians(angulo_de_rotacion))

        if cambio_en_columna > 0:
            cambio_en_columna = cambio_en_columna
            cambio_en_fila = -cambio_en_fila
        
        # Lo hago con el objeto para no meter el codigo de manejo de mascara aca
        degradador_lineas = DegradacionAgregarFrecuenciaUnicaPorPieza(cambio_en_fila, cambio_en_columna, amplitud=self.amplitud)
        resultado, nueva_mascara = degradador_lineas(pieza_rotada, indice, generador, mascara_rotada)

        self.parametros_por_pieza[int(indice)] = {
                            "rotacion": self.degradador_por_rotacion.parametros_por_pieza[int(indice)],
                            "lineas":  degradador_lineas.parametros_por_pieza[int(indice)]
                        }

        return resultado, nueva_mascara


# endregion

def componer_degradaciones(*degradaciones):
    def degradacion_compuesta(imagen, generador):
        res = imagen
        for deg in degradaciones:
            res = deg(res, generador)
        return res
    return degradacion_compuesta

