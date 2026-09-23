from __future__ import annotations

import numpy as np
from numpy.random import Generator
from typing import Any, Optional, TYPE_CHECKING, Union
from core.tipos import RangoFlotante
from abc import ABC, abstractmethod
from core.degradaciones import (
    _acotar_rango,
    agregar_ruido_gaussiano,
    agregar_ruido_uniforme,
    agregar_ruido_rayleigh,
    agregar_ruido_exponencial,
    agregar_ruido_sal_y_pimienta,
    alterar_luminancia,
    alterar_crominancia
)

# Import solo para chequeo de tipos: RompecabezasV2 importa este modulo, asi que
# importar Pieza en tiempo de ejecucion crearia un ciclo.
if TYPE_CHECKING:
    from core.RompecabezasV2 import Pieza

# region Degradadores Globales
class DegradadorGlobal(ABC):

    @abstractmethod
    def _aplicar_degradacion(self, imagen:np.ndarray, generador : np.random.Generator) -> np.ndarray:
        pass

    def aplicar_degradacion(self, imagen:np.ndarray, generador : np.random.Generator) -> np.ndarray:
        degradacion = self._aplicar_degradacion(imagen, generador)
        return _acotar_rango(degradacion)

class DegradadorGaussiano(DegradadorGlobal):

    def __init__(self, desviacion_estandar: float = 0.08, media: float = 0.0):
        self.desviacion_estandar =  desviacion_estandar
        self.media = media
    
    def _aplicar_degradacion(self, imagen: np.ndarray, generador: np.random.Generator) -> np.ndarray:
        return agregar_ruido_gaussiano(imagen, generador, self.desviacion_estandar, self.media)

class DegradadorUniforme(DegradadorGlobal):

    def __init__(self, limite_inferior: float = 0.08, limite_superior: float = 0.0):
        self.limite_inferior =  limite_inferior
        self.limite_superior =  limite_superior
    
    def _aplicar_degradacion(self, imagen: np.ndarray, generador: np.random.Generator) -> np.ndarray:
        return agregar_ruido_uniforme(imagen, generador, self.limite_inferior, self.limite_superior)

class DegradadorRayleigh(DegradadorGlobal):

    def __init__(self, desplazamiento: float = 0.0, parametro_b: float = 0.015, centrar_media: bool = True):
        self.desplazamiento = desplazamiento
        self.parametro_b = parametro_b
        self.centrar_media = centrar_media

    def _aplicar_degradacion(self, imagen: np.ndarray, generador: np.random.Generator) -> np.ndarray:
        return agregar_ruido_rayleigh(imagen, generador, self.desplazamiento, self.parametro_b, self.centrar_media)

class DegradadorExponencial(DegradadorGlobal):

    def __init__(self, tasa_a: float = 15.0, centrar_media: bool = True):
        self.tasa_a = tasa_a
        self.centrar_media = centrar_media

    def _aplicar_degradacion(self, imagen: np.ndarray, generador: np.random.Generator) -> np.ndarray:
        return agregar_ruido_exponencial(imagen, generador, self.tasa_a, self.centrar_media)

class DegradadorSalYPimienta(DegradadorGlobal):

    def __init__(self, probabilidad_sal: float = 0.02, probabilidad_pimienta: float = 0.02):
        self.probabilidad_sal = probabilidad_sal
        self.probabilidad_pimienta = probabilidad_pimienta

    def _aplicar_degradacion(self, imagen: np.ndarray, generador: np.random.Generator) -> np.ndarray:
        return agregar_ruido_sal_y_pimienta(imagen, generador, self.probabilidad_sal, self.probabilidad_pimienta)

class DegradadorAleatorioGlobal(DegradadorGlobal):

    def __init__(self, *degradadores: DegradadorGlobal):
        self.degradadores = degradadores

    def __elegir_degradador_aleatorio(self, generador:np.random.Generator) -> DegradadorGlobal:
        degradadores = list(self.degradadores)
        idx = generador.integers(0, len(degradadores))
        return degradadores[idx]

    def _aplicar_degradacion(self, imagen: np.ndarray, generador: Generator) -> np.ndarray:
        degradador = self.__elegir_degradador_aleatorio(generador)
        return degradador.aplicar_degradacion(imagen, generador)

class DegradadorCompuesto(DegradadorGlobal):

    def __init__(self, *degradadores: DegradadorGlobal):
        self.degradadores = degradadores

    def _aplicar_degradacion(self, imagen: np.ndarray, generador: Generator) -> np.ndarray:

        imagen_degradada = imagen 
        for degradador in self.degradadores:
            imagen_degradada = degradador.aplicar_degradacion(imagen_degradada, generador)

        return imagen_degradada

# endregion

# Cosa Loquis Loquis, me ahorro igual lineas de codigo que las que use para escribirla
def _sortear_argumentos(generador:np.random.Generator, *argumentos:Union[RangoFlotante, float]) -> tuple[float, ...]:
    sorteo = []
    for argumento in argumentos:
        if isinstance(argumento, tuple):
            valor_sorteado = generador.uniform(*argumento)
            sorteo.append(valor_sorteado)
        elif isinstance(argumento, float):
            sorteo.append(argumento)
        else:
            raise ValueError("Tipo Invalido")
    return tuple(sorteo)

class DegradadorPorPieza(ABC):

    def __init__(self):
        self.argumentos_por_pieza: dict[int, dict[str, Any]] = {}

    def resetear(self):
        self.argumentos_por_pieza.clear()

    @abstractmethod
    def _aplicar_degradacion(self, pieza: Pieza, indice: int, generador: np.random.Generator):
        pass

    def aplicar_degradacion(self, pieza: Pieza, indice: int, generador: np.random.Generator):
            if pieza.id in self.argumentos_por_pieza.keys():
                print("Se esta aplicando la degradacion dos veces a la misma pieza!!!")

            self._aplicar_degradacion(pieza, indice, generador)
            self.argumentos_por_pieza[pieza.id]["indice"] = indice

    @property
    @abstractmethod
    def nombre(self) -> str:
        pass

# region Degradadores Nivel 2

class DegradadorPorLuminancia(DegradadorPorPieza):

    def __init__(self, gamma:Union[RangoFlotante, float]=1.0, contraste:Union[RangoFlotante,float]=1.0, brillo:Union[RangoFlotante,float]=0.0):
        super().__init__()
        self.gamma= gamma
        self.contraste= contraste
        self.brillo= brillo

    @property
    def nombre(self) -> str:
        return "luminancia"

    def _aplicar_degradacion(self, pieza: Pieza, indice: int, generador: np.random.Generator):

        imagen_pieza = pieza.imagen.copy()
        gamma, contraste, brillo = _sortear_argumentos(generador, self.gamma, self.contraste, self.brillo)
        imagen_pieza_degradada = alterar_luminancia(imagen_pieza, gamma, contraste, brillo)

        # No toca la mascara
        pieza.actualizar_imagen(imagen_pieza_degradada)

        self.argumentos_por_pieza[pieza.id] = {
            "gamma": gamma,
            "contraste": contraste,
            "brillo": brillo
        }


class DegradadorPorCrominancia(DegradadorPorPieza):

    def __init__(self, ganancia_cb:Union[RangoFlotante, float]=1.0, ganancia_cr:Union[RangoFlotante,float]=1.0, desplazamiento_cb:Union[RangoFlotante,float]=0.0, desplazamiento_cr:Union[RangoFlotante,float]=0.0):
        super().__init__()
        self.ganancia_cb= ganancia_cb
        self.ganancia_cr= ganancia_cr
        self.desplazamiento_cb= desplazamiento_cb
        self.desplazamiento_cr= desplazamiento_cr

    @property
    def nombre(self) -> str:
        return "crominancia"

    def _aplicar_degradacion(self, pieza: Pieza, indice: int, generador: np.random.Generator):

        imagen_pieza = pieza.imagen.copy()
        ganancia_cb, ganancia_cr, desplazamiento_cb, desplazamiento_cr = _sortear_argumentos(generador, self.ganancia_cb, self.ganancia_cr, self.desplazamiento_cb, self.desplazamiento_cr)
        imagen_pieza_degradada = alterar_crominancia(imagen_pieza, ganancia_cb, ganancia_cr, desplazamiento_cb, desplazamiento_cr)

        # No toca la mascara
        pieza.actualizar_imagen(imagen_pieza_degradada)

        self.argumentos_por_pieza[pieza.id] = {
            "ganancia_cb": ganancia_cb,
            "ganancia_cr": ganancia_cr,
            "desplazamiento_cb": desplazamiento_cb,
            "desplazamiento_cr" : desplazamiento_cr
        }

class DegradadorAleatorio(DegradadorPorPieza):

    def __init__(self, *degradadores: DegradadorPorPieza, asignaciones: Optional[list[str]] = None):
        super().__init__()
        self.degradadores = degradadores
        self.asignaciones = asignaciones
        self.__assert_asignaciones_validas()

    @property
    def lista_degradadores(self) -> list[str]:
        return [degradador.nombre for degradador in self.degradadores]

    @property
    def nombre(self) -> str:
        return f"aleatorio{str(self.lista_degradadores)}"

    def __assert_asignaciones_validas(self):
        if self.asignaciones is not None:
            nombres_validos = {degradador.nombre for degradador in self.degradadores}
            for asignacion in self.asignaciones:
                assert asignacion in nombres_validos, \
                    f"Asignacion invalida: {asignacion!r}. Se esperaba uno de {nombres_validos}"

    def __encontrar_degradador_por_nombre(self, nombre:str) -> DegradadorPorPieza:
        assert self.asignaciones is not None, "No se puede buscar en una asignacion no existente"

        # Python es feo
        return next(filter(lambda degradador: degradador.nombre == nombre, self.degradadores))

    def __elegir_degradador_aleatorio(self, generador:np.random.Generator) -> DegradadorPorPieza:
        degradadores = list(self.degradadores)
        idx = generador.integers(0, len(degradadores))
        return degradadores[idx]

    def _aplicar_degradacion(self, pieza: Pieza, indice: int, generador: np.random.Generator):

        if self.asignaciones:
            indice_degradador = indice % len(self.asignaciones)
            nombre_degradador = self.asignaciones[indice_degradador]
            degradador = self.__encontrar_degradador_por_nombre(nombre_degradador)
        else: 
            degradador = self.__elegir_degradador_aleatorio(generador)

        degradador.aplicar_degradacion(pieza, indice, generador)

        self.argumentos_por_pieza[pieza.id] = {
            "degradador" : degradador.nombre,
            "argumentos_degradador": degradador.argumentos_por_pieza[pieza.id]
        }

# Para Leo
class DegradadorBalanceado(DegradadorAleatorio):

    def __init__(self, *degradadores: DegradadorPorPieza, cantidad_de_piezas: int):

        self.cantidad_piezas = cantidad_de_piezas
        self.degradadores = degradadores

        super().__init__(*degradadores, asignaciones=None)

    @property
    def nombre(self) -> str:
        return f"balanceado{str(self.lista_degradadores)}"

    def generar_asignacion_balanceada(self, generador:np.random.Generator):

        fraccion = self.cantidad_piezas // len(self.degradadores)

        etiquetas = []

        for degradador in self.degradadores[:-1]:
            etiquetas += [degradador.nombre] * fraccion

        ultimo_degradador = self.degradadores[-1]
        asignaciones_faltantes = self.cantidad_piezas - len(etiquetas)
        etiquetas += [ultimo_degradador.nombre] * asignaciones_faltantes

        generador.shuffle(etiquetas)

        return etiquetas

    def _aplicar_degradacion(self, pieza: Pieza, indice: int, generador: Generator):

        if not self.asignaciones:
            self.asignaciones = self.generar_asignacion_balanceada(generador)

        return super()._aplicar_degradacion(pieza, indice, generador)


class DegradadorPorPiezaCompuesto(DegradadorPorPieza):
    """Aplica varios DegradadorPorPieza a la MISMA pieza, en cadena y en el orden dado
    (analogo a DegradadorCompuesto pero para degradaciones por pieza)."""

    def __init__(self, *degradadores: DegradadorPorPieza):
        super().__init__()
        self.degradadores = degradadores

    @property
    def nombre(self) -> str:
        return f"compuesto{[degradador.nombre for degradador in self.degradadores]}"

    def _aplicar_degradacion(self, pieza: Pieza, indice: int, generador: Generator):

        for degradador in self.degradadores:
            degradador._aplicar_degradacion(pieza, indice, generador)

        self.argumentos_por_pieza[pieza.id] = {
            degradador.nombre: degradador.argumentos_por_pieza.get(pieza.id)
            for degradador in self.degradadores
        }


# endregion

# region Degradacion Nivel 3

class DegradadorFourier(DegradadorPorPieza, ABC):
    def __init__(self, frecuencias_disponibles: tuple[int, ...], amplitud: Union[RangoFlotante, float]):
        super().__init__()
        self.frecuencias_disponibles = frecuencias_disponibles
        self.amplitud = amplitud

    @abstractmethod
    def _sortear_frecuencias(self, generador: np.random.Generator, amplitud: float)-> tuple[tuple[int, int, float, float], ...]:
        pass

    @property
    @abstractmethod
    def tipo_trama(self) -> str:
        pass

    @property
    def nombre(self) -> str:
        return f"fourier({self.tipo_trama})"

    def __recolectar_picos(self, ondas_seleccionadas: tuple[tuple[int, int, float, float], ...])-> tuple[tuple[int, int], ...]:
        lista_de_picos = [(frecuencia_horizontal, frecuencia_vertical) for (frecuencia_horizontal, frecuencia_vertical, _, _) in ondas_seleccionadas]
        return tuple(lista_de_picos)

    def _elegir_frecuencia(self, generador: np.random.Generator) -> int:
        """Elige al azar una de las frecuencias disponibles para esta trama."""
        return int(generador.choice(self.frecuencias_disponibles))

    def _frecuencia_diagonal_valida(self, frecuencia: int) -> int:
        """Redondea a un valor par (y no menor a 8) para que las diagonales queden simetricas."""
        return max(8, (frecuencia // 2) * 2)

    def _fase_aleatoria(self, generador: np.random.Generator) -> float:
        return float(generador.uniform(0, 2 * np.pi))

    def _signo_aleatorio(self, generador: np.random.Generator) -> int:
        return int(generador.choice([-1, 1]))

    def __vector_columna_coeficiente_vertical(self, alto:int, desplazamiento_fila: int) -> np.ndarray:
        return desplazamiento_fila * (np.arange(alto)[:, None] / alto)

    def __vector_fila_coeficiente_horizontal(self, ancho:int, desplazamiento_columna: int) -> np.ndarray:
            return desplazamiento_columna * (np.arange(ancho)[None, :] / ancho)

    def __generar_grilla_fase_espacial(self, imagen:np.ndarray, desplazamiento_fila: int, desplazamiento_columna:int)-> np.ndarray:

        alto, ancho = imagen.shape[:2]

        fase_espacial = 2.0 * np.pi * (
                    self.__vector_columna_coeficiente_vertical(alto, desplazamiento_fila) +
                    self.__vector_fila_coeficiente_horizontal(ancho, desplazamiento_columna)
                    )

        return fase_espacial

    def __agregar_onda(self, imagen: np.ndarray, desplazamiento_fila: int, desplazamiento_columna: int, amplitud: float=0.09, fase: float=0.0):

        fase_espacial = self.__generar_grilla_fase_espacial(imagen, desplazamiento_fila, desplazamiento_columna)
        onda = amplitud * np.sin(fase_espacial + fase)

        # agrego una tercera dimension para que el broadcast funcione
        if imagen.ndim == 3:
            onda = onda[:, :, None]

        return _acotar_rango(imagen + onda)

    def __agregar_ondas(self, imagen:np.ndarray, descriptores_de_ondas:tuple[tuple[int, int, float, float], ...]) -> np.ndarray:
        """Aplica varias ondas (una lista de (fila, columna, amplitud, fase)) en cadena.

        Como cada llamada suma sobre el resultado anterior, el efecto final es
        la superposicion de todas las ondas (interferencia).
        """
        resultado = imagen
        for desplazamiento_fila, desplazamiento_columna, amplitud, fase in descriptores_de_ondas:
            resultado = self.__agregar_onda(resultado, desplazamiento_fila,desplazamiento_columna, amplitud, fase)
        return resultado

    def __generar_lista_diccionario_descriptores_onda(self, descriptores_de_ondas: tuple[tuple[int, int, float, float], ...]) -> list[dict[str,Union[float,int]]]:

        lista_descriptores = []

        for (desplazamiento_filas, desplazamiento_columnas, amplitud, fase) in descriptores_de_ondas:

            diccionario_informacion = {
                "desplazamiento_filas" : desplazamiento_filas,
                "desplazamiento_columnas" : desplazamiento_columnas,
                "amplitud": amplitud,
                "fase": fase
            }

            lista_descriptores.append(diccionario_informacion)

        return lista_descriptores

    def _aplicar_degradacion(self, pieza: Pieza, indice: int, generador: np.random.Generator):

        # syntax sugar para usar la funcion, devuelve una tupla siempre
        (amplitud,) = _sortear_argumentos(generador, self.amplitud)

        ondas_seleccionadas = self._sortear_frecuencias(generador, amplitud)

        imagen_pieza = pieza.imagen.copy()
        imagen_pieza_degradada = self.__agregar_ondas(imagen_pieza, ondas_seleccionadas)

        # No toca la mascara
        pieza.actualizar_imagen(imagen_pieza_degradada)

        self.argumentos_por_pieza[pieza.id] = {
            "amplitud": round(amplitud, 4),
            "ondas": self.__generar_lista_diccionario_descriptores_onda(ondas_seleccionadas),
            "picos": self.__recolectar_picos(ondas_seleccionadas),
        }


class DegradadorFourierOrtogonal(DegradadorFourier):
    """Una onda que varia solo por columna (lineas verticales) y otra que
    varia solo por fila (lineas horizontales) -> grilla recta."""

    @property
    def tipo_trama(self) -> str:
        return "ortogonales"

    def _sortear_frecuencias(self, generador: np.random.Generator, amplitud: float):
        frecuencia_lineas_verticales = self._elegir_frecuencia(generador)
        frecuencia_lineas_horizontales = self._elegir_frecuencia(generador)
        return (
            (0, frecuencia_lineas_verticales, amplitud, self._fase_aleatoria(generador)),
            (frecuencia_lineas_horizontales, 0, amplitud, self._fase_aleatoria(generador)),
        )


class DegradadorFourierDiagonal(DegradadorFourier):
    """Misma frecuencia en fila y columna (diagonal "\\") y su espejo con
    columna negada (diagonal "/") -> forman una X."""

    @property
    def tipo_trama(self) -> str:
        return "diagonales"

    def _sortear_frecuencias(self, generador: np.random.Generator, amplitud: float):
        frecuencia_diagonal = self._frecuencia_diagonal_valida(self._elegir_frecuencia(generador))
        return (
            (frecuencia_diagonal, frecuencia_diagonal, amplitud, self._fase_aleatoria(generador)),
            (frecuencia_diagonal, -frecuencia_diagonal, amplitud, self._fase_aleatoria(generador)),
        )


class DegradadorFourierOblicuo(DegradadorFourier):
    """Dos ondas donde se cruzan fila/columna entre si (una usa la
    frecuencia de la otra), por eso las lineas quedan en angulos oblicuos
    en vez de horizontales/verticales/diagonales puras."""

    @property
    def tipo_trama(self) -> str:
        return "oblicuas"

    def _sortear_frecuencias(self, generador: np.random.Generator, amplitud: float):
        frecuencia_base = self._elegir_frecuencia(generador)
        frecuencia_secundaria = self._elegir_frecuencia(generador)
        frecuencia_oblicua = max(8, frecuencia_base // 2)
        return (
            (frecuencia_oblicua, frecuencia_secundaria, amplitud, self._fase_aleatoria(generador)),
            (frecuencia_secundaria, self._signo_aleatorio(generador) * frecuencia_oblicua, amplitud, self._fase_aleatoria(generador)),
        )


class DegradadorFourierTriple(DegradadorFourier):
    """Horizontal + vertical + diagonal combinadas, cada una mas tenue (80%
    de la amplitud) para que la suma de las tres no sature."""

    @property
    def tipo_trama(self) -> str:
        return "triple"

    def _sortear_frecuencias(self, generador: np.random.Generator, amplitud: float):
        frecuencia_vertical = self._elegir_frecuencia(generador)
        frecuencia_horizontal = self._elegir_frecuencia(generador)
        frecuencia_diagonal = self._frecuencia_diagonal_valida(self._elegir_frecuencia(generador))
        amplitud_parcial = amplitud * 0.8
        return (
            (0, frecuencia_vertical, amplitud_parcial, self._fase_aleatoria(generador)),
            (frecuencia_horizontal, 0, amplitud_parcial, self._fase_aleatoria(generador)),
            (frecuencia_diagonal, frecuencia_diagonal, amplitud_parcial, self._fase_aleatoria(generador)),
        )


class DegradadorFourierDobleFrecuencia(DegradadorFourier):
    """Dos frecuencias distintas sobre el mismo eje (ambas horizontales o
    ambas verticales) -> genera un patron de interferencia ("latido") a lo
    largo de ese eje."""

    @property
    def tipo_trama(self) -> str:
        return "doble_frecuencia"

    def __elegir_par_de_frecuencias_ordenadas(self, generador: np.random.Generator) -> tuple[int, int]:
        
        frecuencias_candidatas_ordenadas = sorted({self._elegir_frecuencia(generador) for _ in range(3)})
        frecuencia_baja, frecuencia_alta = frecuencias_candidatas_ordenadas[0], frecuencias_candidatas_ordenadas[-1]

        if frecuencia_baja == frecuencia_alta:
            frecuencia_baja = min(self.frecuencias_disponibles)
            frecuencia_alta = max(self.frecuencias_disponibles)

        return frecuencia_baja, frecuencia_alta

    def _sortear_frecuencias(self, generador: np.random.Generator, amplitud: float):

        frecuencia_baja, frecuencia_alta = self.__elegir_par_de_frecuencias_ordenadas(generador)

        eje_elegido = str(generador.choice(["horizontal", "vertical"]))

        if eje_elegido == "horizontal":
            return (
                (0, frecuencia_baja, amplitud, self._fase_aleatoria(generador)),
                (0, frecuencia_alta, amplitud, self._fase_aleatoria(generador)),
                )
        else:
            return (
                (frecuencia_baja, 0, amplitud, self._fase_aleatoria(generador)),
                (frecuencia_alta, 0, amplitud, self._fase_aleatoria(generador)),
            )


class DegradadorFourierCuadruple(DegradadorFourier):
    """Horizontal + vertical + las dos diagonales juntas, cada una mas
    tenue (65% de la amplitud): es la trama mas densa/destructiva."""

    @property
    def tipo_trama(self) -> str:
        return "cuadruple"

    def _sortear_frecuencias(self, generador: np.random.Generator, amplitud: float):
        frecuencia_vertical = self._elegir_frecuencia(generador)
        frecuencia_horizontal = self._elegir_frecuencia(generador)
        frecuencia_diagonal = self._frecuencia_diagonal_valida(min(frecuencia_vertical, frecuencia_horizontal))
        amplitud_parcial = amplitud * 0.65
        return (
            (0, frecuencia_vertical, amplitud_parcial, self._fase_aleatoria(generador)),
            (frecuencia_horizontal, 0, amplitud_parcial, self._fase_aleatoria(generador)),
            (frecuencia_diagonal, frecuencia_diagonal, amplitud_parcial, self._fase_aleatoria(generador)),
            (frecuencia_diagonal, -frecuencia_diagonal, amplitud_parcial, self._fase_aleatoria(generador)),
        )

# endregion

# region Degradacion Nivel 5

class DegradadorFrecuenciaDeterministica(DegradadorFourier):

    def __init__(self, desplazamiento_fila, desplazamiento_columna, amplitud:float=10):

        self.desplazamiento_fila = desplazamiento_fila
        self.desplazamiento_columna = desplazamiento_columna
        self.amplitud = amplitud

        # Valor Aleatorio, no lo vamos a usar
        super().__init__((42,), amplitud)

    @property
    def tipo_trama(self) -> str:
        return f"deterministica({str(self.desplazamiento_fila)}, {str(self.desplazamiento_columna)})"

    def _sortear_frecuencias(self, generador: np.random.Generator, amplitud: float):
            return (
                (self.desplazamiento_fila, self.desplazamiento_columna, amplitud, self._fase_aleatoria(generador)),
            )

class DegradadorRotacion(DegradadorPorPieza):
    def __init__(self, angulo):
        super().__init__()
        self.angulo = angulo

    @property
    def nombre(self) -> str:
        return "rotar"

    def _aplicar_degradacion(self, pieza: Pieza, indice: int, generador: Generator):

        (angulo,) = _sortear_argumentos(generador, self.angulo)

        pieza.rotar(angulo)

        self.argumentos_por_pieza[pieza.id] = {
            "angulo" : angulo
        }

# Composicion de Rotar y agregar lineas para que el angulo de las lineas y rotacion sean el mismo
class DegradadorRotacionYLineasAlMismoAngulo(DegradadorRotacion):
    def __init__(self, angulo:Union[RangoFlotante, float], amplitud:float=0.3, cantidad_lineas:int= 50):
        super().__init__(angulo)

        # Dejo todo en nulo, lo voy a setear por aplicacion
        self.degradadorFrecuenciaDeterministica = DegradadorFrecuenciaDeterministica(0, 0, amplitud)
        self.cantidad_lineas= cantidad_lineas

    @property
    def nombre(self) -> str:
        return "rotar_y_agregar_lineas"

    def __actualizar_degradador_lineas_a_nuevo_angulo(self, angulo):

        # Probe combinaciones hasta que funciono, puede ser que sea equivalente a algo
        # donde fila sea seno y columna sea coseno
        # da igual la verdad.
        cambio_en_columna = self.cantidad_lineas * np.sin(np.radians(angulo))
        cambio_en_fila = self.cantidad_lineas * np.cos(np.radians(angulo))

        if cambio_en_columna > 0:
            cambio_en_columna = cambio_en_columna
            cambio_en_fila = -cambio_en_fila

        self.degradadorFrecuenciaDeterministica.desplazamiento_fila = cambio_en_fila
        self.degradadorFrecuenciaDeterministica.desplazamiento_columna = cambio_en_columna

    def _aplicar_degradacion(self, pieza: Pieza, indice: int, generador: Generator):

        super()._aplicar_degradacion(pieza, indice, generador)

        angulo = self.argumentos_por_pieza[pieza.id]["angulo"]

        self.__actualizar_degradador_lineas_a_nuevo_angulo(angulo)
        
        self.degradadorFrecuenciaDeterministica._aplicar_degradacion(pieza, indice, generador)

        self.argumentos_por_pieza[pieza.id].update({
            "cantidad_lineas": self.cantidad_lineas,
            "degradador_lineas": self.degradadorFrecuenciaDeterministica.argumentos_por_pieza[pieza.id]
        })

# endregion


def reiniciar_estado_degradador_por_pieza(degradador: Optional[DegradadorPorPieza]) -> None:
    """Limpia el estado acumulado de un DegradadorPorPieza (argumentos_por_pieza, y la
    asignacion balanceada de DegradadorBalanceado) para poder reusar la MISMA instancia en
    un RompecabezasV2 nuevo. Los degradadores de core/degradadores_secciones.py son
    instancias unicas a nivel de modulo, pensadas para un solo rompecabezas por notebook;
    al recorrer muchas imagenes con la misma instancia hay que reiniciarlas entre corridas,
    sino ids de pieza repetidos (0..N-1 de cada rompecabezas nuevo) disparan el aviso de
    "se esta aplicando la degradacion dos veces" y, en DegradadorBalanceado, la asignacion
    queda congelada en la del primer rompecabezas. Recorre tambien los degradadores anidados
    (DegradadorAleatorio/Balanceado/Compuesto/RotacionYLineas)."""
    if degradador is None:
        return

    degradador.resetear()
    if hasattr(degradador, "asignaciones"):
        degradador.asignaciones = None

    for sub_degradador in getattr(degradador, "degradadores", ()):
        reiniciar_estado_degradador_por_pieza(sub_degradador)

    sub_degradador_lineas = getattr(degradador, "degradadorFrecuenciaDeterministica", None)
    if sub_degradador_lineas is not None:
        reiniciar_estado_degradador_por_pieza(sub_degradador_lineas)