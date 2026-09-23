from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import numpy as np
from core.detector_forma import generar_mascara_de_pieza
import cv2
from core.crear_rompecabezas import pre_proceso_imagen_para_rompecabezas, cortar_piezas
from core.analizador_rotacion import enderezar_pieza
from core.degradacionesV2 import DegradadorPorPieza, DegradadorGlobal
from core.tipos import BoolArray
from core.geometria_jigsaw import JigsawGridGeometry, DescriptorGeometria
from core.detector_forma import detect_corners_and_split_sides, extraer_contorno_externo,pasar_borde_a_1d

class RompecabezasV2:

    def __init__(
        self,
        imagen: np.ndarray,
        cantidad_filas: int,
        cantidad_columnas: int,
        tiene_ranuras: bool = False,
        degradador_global: Optional[DegradadorGlobal] = None,
        degradador_por_pieza: Optional[DegradadorPorPieza] = None,
        semilla: int = 42,
        padding: int = 30,
        metadatos: Dict[str, Any] = {},
    ):
        self.cantidad_filas = cantidad_filas
        self.cantidad_columnas = cantidad_columnas
        self.tiene_ranuras = tiene_ranuras
        self.metadatos = metadatos
        self.semilla = semilla
        self.padding = padding

        rng = np.random.default_rng(semilla)

        img_ajustada = pre_proceso_imagen_para_rompecabezas(imagen, cantidad_filas, cantidad_columnas)

        if degradador_global:
            img_ajustada = degradador_global.aplicar_degradacion(img_ajustada, rng) 

        self.imagen = img_ajustada
        self.piezas = self.__cortar_piezas()

        if degradador_por_pieza:
            self.__aplicar_degradacion_por_piezas(degradador_por_pieza, rng)

        self.estado_ordenado = self.__generar_estado_ordenado()
        self.imagen = self.pegar_piezas()

        self.metadatos = {
            "imagen": img_ajustada,
            "semilla": semilla,
            "filas": cantidad_filas,
            "columnas": cantidad_columnas,
        }

        if metadatos:
            self.metadatos.update(metadatos)

    def __generar_estado_ordenado(self):
        return np.arange(self.cantidad_filas * self.cantidad_columnas).reshape(self.cantidad_filas, self.cantidad_columnas)

    def __aplicar_degradacion_por_piezas(self, degradador:DegradadorPorPieza, rng: np.random.Generator):

        for idx, pieza in enumerate(self.piezas):
            degradador.aplicar_degradacion(pieza, idx, rng)

    def __conseguir_maxima_desviacion_de_borde(self, puntos_borde:DescriptorGeometria):
        maxima_desviacion_norte = pasar_borde_a_1d(puntos_borde.borde_norte, "NORTE")["max_dev"]
        maxima_desviacion_este = pasar_borde_a_1d(puntos_borde.borde_este, "ESTE")["max_dev"]
        maxima_desviacion_sur = pasar_borde_a_1d(puntos_borde.borde_sur, "SUR")["max_dev"]
        maxima_desviacion_oeste = pasar_borde_a_1d(puntos_borde.borde_oeste, "OESTE")["max_dev"]

        return maxima_desviacion_norte, maxima_desviacion_este, maxima_desviacion_sur, maxima_desviacion_oeste

    def __cortar_piezas_por_geometria(self, jigsaw:JigsawGridGeometry, padding=30) -> Tuple[Pieza, ...]:

        alto, ancho = self.imagen.shape[:2]

        alto_pieza = alto // self.cantidad_filas
        ancho_pieza = ancho // self.cantidad_columnas

        pad_width = ((padding, padding), (padding, padding), (0, 0)) if self.imagen.ndim == 3 else padding
        imagen_con_padding = np.pad(self.imagen, pad_width)

        piezas = []
        for fila in range(self.cantidad_filas):
            for columna in range(self.cantidad_columnas):

                index = np.ravel_multi_index((fila, columna), (self.cantidad_filas, self.cantidad_columnas))
                
                geometria = jigsaw.extraer_geometria_por_pieza(fila,columna)
                desviacion_norte, desviacion_este, desviacion_sur, desviacion_oeste = self.__conseguir_maxima_desviacion_de_borde(geometria)
                desviacion_norte, desviacion_este, desviacion_sur, desviacion_oeste = (
                    int(round(desviacion_norte)), int(round(desviacion_este)),
                    int(round(desviacion_sur)), int(round(desviacion_oeste)),
                )

                y_inicio_pieza = fila * alto_pieza - desviacion_norte
                y_fin_pieza = y_inicio_pieza + alto_pieza + padding * 2 + desviacion_norte + desviacion_sur

                x_inicio_pieza = columna * ancho_pieza - desviacion_oeste
                x_fin_pieza = x_inicio_pieza + ancho_pieza + padding * 2 + desviacion_oeste + desviacion_este
                
                parche_pieza = imagen_con_padding[y_inicio_pieza:y_fin_pieza, x_inicio_pieza:x_fin_pieza].copy()
                piezas.append(Pieza(parche_pieza,geometria,self,int(index),padding))

        return tuple(piezas) 

    def __cortar_piezas(self) -> Tuple[Pieza, ...]:

        altura, ancho = self.imagen.shape[:2]

        if self.tiene_ranuras:
            jigsaw = JigsawGridGeometry(self.cantidad_filas, self.cantidad_columnas, altura, ancho, seed=self.semilla, discrete=True)
        else:
            jigsaw = JigsawGridGeometry(self.cantidad_filas, self.cantidad_columnas, altura, ancho, profile_types=["borde"], seed=self.semilla)

        return self.__cortar_piezas_por_geometria(jigsaw)

    @property
    def cantidad_piezas(self) -> int:
        return len(self.piezas)

    def obtener_matriz_correcta(self) -> np.ndarray:
        """Matriz 2D solución donde grilla[r, c] = id_pieza."""
        return self.estado_ordenado.copy()

    def barajar_piezas(self, generador: Optional[np.random.Generator] = None) -> List[Pieza]:
        """Lista con las Piezas (no sus ids) en orden al azar."""
        if generador is None:
            generador = np.random.default_rng()

        ids_barajados = generador.permutation(self.cantidad_piezas)
        return [self.piezas[id_pieza] for id_pieza in ids_barajados]

    def obtener_adyacencias_correctas(self) -> Dict[str, Dict[int, int]]:
        """
        Retorna las adyacencias verdaderas:
        - 'horizontal': {id_izq: id_der}
        - 'vertical': {id_arriba: id_abajo}
        """
        return {
            "horizontal": {int(self.estado_ordenado[r, c]): int(self.estado_ordenado[r, c + 1])
                           for r in range(self.cantidad_filas)
                           for c in range(self.cantidad_columnas - 1)},
            "vertical": {int(self.estado_ordenado[r, c]): int(self.estado_ordenado[r + 1, c])
                         for r in range(self.cantidad_filas - 1)
                         for c in range(self.cantidad_columnas)},
        }

    def pegar_piezas(
        self,
        grilla_propuesta: Optional[np.ndarray] = None,
    ) -> np.ndarray:

        if grilla_propuesta is None:
            grilla_propuesta = self.obtener_matriz_correcta()

        grilla_arr = np.asarray(grilla_propuesta)

        alto, ancho = self.imagen.shape[:2]
        alto_pieza = alto // self.cantidad_filas
        ancho_pieza = ancho // self.cantidad_columnas
        canales = self.imagen.shape[2] if self.imagen.ndim == 3 else 1

        lienzo = np.zeros((alto, ancho, canales), dtype=np.float64)

        for fila in range(self.cantidad_filas):
            for columna in range(self.cantidad_columnas):
                id_pieza = int(grilla_arr[fila, columna])
                if 0 <= id_pieza < len(self.piezas):
                    self.__pegar_pieza_en_lienzo(lienzo, self.piezas[id_pieza], fila, columna, alto_pieza, ancho_pieza)

        lienzo = self.__rellenar_huecos_por_inpainting(lienzo)

        if canales == 1:
            lienzo = lienzo.squeeze(axis=-1)
        return np.clip(lienzo, 0.0, 1.0)

    def __pegar_pieza_en_lienzo(
        self,
        lienzo: np.ndarray,
        pieza: Pieza,
        fila: int,
        columna: int,
        alto_pieza: int,
        ancho_pieza: int,
    ) -> None:

        parche = pieza.devolver_pieza()
        mascara_parche = parche[..., -1] > 0
        color_parche = parche[..., :-1]

        x_inicial, _, y_inicial, _ = pieza.geometria.bounding_box_real()

        y_inicio_lienzo = fila * alto_pieza - int(y_inicial)
        x_inicio_lienzo = columna * ancho_pieza - int(x_inicial)

        alto_lienzo, ancho_lienzo = lienzo.shape[:2]
        y_inicio_valido = max(y_inicio_lienzo, 0)
        x_inicio_valido = max(x_inicio_lienzo, 0)
        y_fin_valido = min(y_inicio_lienzo + parche.shape[0], alto_lienzo)
        x_fin_valido = min(x_inicio_lienzo + parche.shape[1], ancho_lienzo)

        if y_fin_valido <= y_inicio_valido or x_fin_valido <= x_inicio_valido:
            return

        y_inicio_en_parche, x_inicio_en_parche = y_inicio_valido - y_inicio_lienzo, x_inicio_valido - x_inicio_lienzo
        y_fin_en_parche = y_inicio_en_parche + (y_fin_valido - y_inicio_valido)
        x_fin_en_parche = x_inicio_en_parche + (x_fin_valido - x_inicio_valido)

        region_mascara = mascara_parche[y_inicio_en_parche:y_fin_en_parche, x_inicio_en_parche:x_fin_en_parche, None]
        region_color = color_parche[y_inicio_en_parche:y_fin_en_parche, x_inicio_en_parche:x_fin_en_parche]

        destino = lienzo[y_inicio_valido:y_fin_valido, x_inicio_valido:x_fin_valido]
        lienzo[y_inicio_valido:y_fin_valido, x_inicio_valido:x_fin_valido] = np.where(region_mascara, region_color, destino)

    def __rellenar_huecos_por_inpainting(self, lienzo: np.ndarray) -> np.ndarray:

        canales = lienzo.shape[2]
        mascara_negra = np.all(lienzo < 0.1, axis=-1) if canales == 3 else (lienzo < 0.1)
        pct_negro = float(np.mean(mascara_negra))

        if not (0 < pct_negro < 0.20):
            return lienzo

        mask_inpaint = mascara_negra.astype(np.uint8) * 255
        lienzo_u8 = np.clip(lienzo * 255.0, 0, 255).astype(np.uint8)
        inpa_u8 = cv2.inpaint(lienzo_u8, mask_inpaint, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

        mask_rem = ((np.all(inpa_u8 == 0, axis=-1) if canales == 3 else (inpa_u8 == 0)) & (mask_inpaint > 0)).astype(np.uint8) * 255
        if np.any(mask_rem > 0):
            inpa_u8 = cv2.inpaint(inpa_u8, mask_rem, inpaintRadius=7, flags=cv2.INPAINT_TELEA)

        return inpa_u8.astype(np.float64) / 255.0

class MascaraMuesca:
    def __init__(self, descriptor_geometria: DescriptorGeometria):
        self.borde_norte = descriptor_geometria.borde_norte
        self.borde_sur = descriptor_geometria.borde_sur
        self.borde_oeste = descriptor_geometria.borde_oeste
        self.borde_este = descriptor_geometria.borde_este
        self.dimensiones_base = descriptor_geometria.dimensiones_base 
        self.punto_inicial = descriptor_geometria.punto_inicial
        self.angulo = 0
        self.__relativilizar_puntos()

        x_inicio, x_fin, y_inicio, y_fin = self.bounding_box_real()
        self.puntos_esquinas = np.asarray([[x_inicio, y_inicio], [x_fin, y_inicio], [x_fin, y_fin], [x_inicio, y_fin]])

    def __poligono_geometrico(self) -> np.ndarray:
        # Doy vuelta sur y oeste para que empiezen de deracha a izquierda y de abajo hacia arriba
        return np.vstack([
            self.borde_norte.astype(np.int32),
            self.borde_este.astype(np.int32),
            np.flip(self.borde_sur, axis=0).astype(np.int32),
            np.flip(self.borde_oeste, axis=0).astype(np.int32),
        ])

    def __bounding_box_poligono(self):
        poligono_silueta = self.__poligono_geometrico()
        x_inicio_silueta = int(np.min(poligono_silueta[:, 0]))
        x_fin_silueta = int(np.max(poligono_silueta[:, 0])) + 1

        y_inicio_silueta = int(np.min(poligono_silueta[:, 1]))
        y_fin_silueta = int(np.max(poligono_silueta[:, 1])) + 1

        return x_inicio_silueta, x_fin_silueta, y_inicio_silueta, y_fin_silueta

    def __dimensiones_poligono(self):
        x_inicio_silueta, x_fin_silueta, y_inicio_silueta, y_fin_silueta = self.__bounding_box_poligono()
        ancho_silueta = x_fin_silueta - x_inicio_silueta
        alto_silueta = y_fin_silueta - y_inicio_silueta

        return ancho_silueta, alto_silueta
    
    def __relativilizar_puntos_de_borde(self, puntos_borde:np.ndarray, x_inicio, y_inicio):
        for punto in puntos_borde:
            punto[0] -= x_inicio
            punto[1] -= y_inicio

    def __relativilizar_puntos(self):
        x_inicio_silueta, _, y_inicio_silueta, _ = self.__bounding_box_poligono()
        self.__relativilizar_puntos_de_borde(self.borde_norte, x_inicio_silueta, y_inicio_silueta)
        self.__relativilizar_puntos_de_borde(self.borde_sur, x_inicio_silueta, y_inicio_silueta)
        self.__relativilizar_puntos_de_borde(self.borde_oeste, x_inicio_silueta, y_inicio_silueta)
        self.__relativilizar_puntos_de_borde(self.borde_este, x_inicio_silueta, y_inicio_silueta)

        self.punto_inicial[0] -= x_inicio_silueta
        self.punto_inicial[1] -= y_inicio_silueta

    def desviacion_borde(self, borde:str)->np.ndarray:
        lado_u = borde.upper()
        if lado_u == "NORTE":
            eje_de_movimento = np.array([0.0, -1.0])
            silueta_de_borde = self.borde_norte
        elif lado_u == "SUR":
            eje_de_movimento = np.array([0.0, 1.0])
            silueta_de_borde = self.borde_sur
        elif lado_u == "OESTE":
            eje_de_movimento = np.array([-1.0, 0.0])
            silueta_de_borde = self.borde_oeste
        elif lado_u == "ESTE":
            eje_de_movimento = np.array([1.0, 0.0])
            silueta_de_borde = self.borde_este
        else:
            raise ValueError("Nombre de Borde no Valido")

        # Restamos el primer punto (una esquina, siempre en la posicion nominal sin joroba)
        # para medir el desvio relativo al borde plano, no la posicion absoluta del punto.
        silueta_relativa = silueta_de_borde - silueta_de_borde[0]
        return np.dot(silueta_relativa, eje_de_movimento)

    def maxima_desviacion_borde(self, borde:str)-> float:
        desviacion = self.desviacion_borde(borde)
        return np.max(np.abs(desviacion))

    def generar_mascara(self, padding:int = 0)-> BoolArray:
        ancho_silueta, alto_silueta = self.__dimensiones_poligono()
        fondo = np.zeros((alto_silueta, ancho_silueta))
        mascara_base = cv2.fillPoly(fondo, [self.__poligono_geometrico()], 255)
        mascara_base = np.pad(mascara_base, padding)
        mascara_base = mascara_base > 0
        return mascara_base

    def bounding_box_real(self) -> tuple[int, int, int, int]:

        x_inicial = self.punto_inicial[0]
        x_final = self.punto_inicial[0] + self.dimensiones_base[0]
        y_inicial = self.punto_inicial[1]
        y_final = self.punto_inicial[1] + self.dimensiones_base[1]

        return (x_inicial, x_final, y_inicial, y_final)

    def generar_mascara_centrada_en_centro_real(self, padding:int = 0) -> BoolArray:
        # A diferencia de generar_mascara, acomoda el centro real (sin sesgo de las muescas)
        # exactamente en el centro del lienzo, para que sirva de pivote consistente al rotar.
        mascara_ajustada_a_silueta = self.generar_mascara()
        alto_silueta, ancho_silueta = mascara_ajustada_a_silueta.shape

        x_inicial, x_final, y_inicial, y_final = self.bounding_box_real()
        centro_real_x = (x_inicial + x_final) / 2
        centro_real_y = (y_inicial + y_final) / 2

        medio_ancho = max(centro_real_x, ancho_silueta - centro_real_x)
        medio_alto = max(centro_real_y, alto_silueta - centro_real_y)

        x_destino = int(round(medio_ancho - centro_real_x))
        y_destino = int(round(medio_alto - centro_real_y))
        ancho_centrado = int(round(2 * medio_ancho))
        alto_centrado = int(round(2 * medio_alto))

        mascara_centrada = np.zeros((alto_centrado, ancho_centrado), dtype=bool)
        mascara_centrada[y_destino:y_destino + alto_silueta, x_destino:x_destino + ancho_silueta] = mascara_ajustada_a_silueta

        return np.pad(mascara_centrada, padding)

    def __convertir_mascara_booleana_en_numerica(self, mascara:BoolArray)->np.ndarray:
        # suponesmos 255
        return mascara.astype(np.uint8) * 255

    def __convertir_mascara_numerica_en_booleana(self, mascara:np.ndarray)->np.ndarray:
        # suponesmos 255
        return mascara > 100

    def __definir_silueta_sobre_desvios_profundidad(self, desvios:np.ndarray, borde:str)->np.ndarray:

        esquina_tl, esquina_tr, esquina_br, esquina_bl = self.puntos_esquinas

        # Cada lado va de su propia esquina de inicio a su propia esquina de fin (antes
        # se asumia horizontal/vertical en el mundo, solo valido si la pieza no rotaba).
        if borde == "NORTE":
            punto_inicial, punto_final = esquina_tl, esquina_tr
        elif borde == "SUR":
            punto_inicial, punto_final = esquina_bl, esquina_br
        elif borde == "ESTE":
            punto_inicial, punto_final = esquina_tr, esquina_br
        elif borde == "OESTE":
            punto_inicial, punto_final = esquina_tl, esquina_bl
        else:
            raise ValueError("Borde no Valido")

        punto_inicial = np.asarray(punto_inicial, dtype=np.float64)
        punto_final = np.asarray(punto_final, dtype=np.float64)

        vector_recorrido = punto_final - punto_inicial
        longitud_recorrido = np.linalg.norm(vector_recorrido)
        direccion_movimiento = vector_recorrido / longitud_recorrido if longitud_recorrido > 1e-6 else np.array([1.0, 0.0])

        # Las dos perpendiculares posibles; nos quedamos con la que apunta hacia
        # afuera del centro de la pieza (alejandose, no hacia adentro).
        perpendicular = np.array([-direccion_movimiento[1], direccion_movimiento[0]])
        centro_pieza = np.mean(self.puntos_esquinas, axis=0)
        punto_medio_lado = (punto_inicial + punto_final) / 2
        if np.dot(perpendicular, punto_medio_lado - centro_pieza) < 0:
            perpendicular = -perpendicular
        direccion_desvio = perpendicular

        puntos_silueta = []
        for idx, desvio in enumerate(desvios):
            desvio_profundidad = direccion_desvio * desvio
            avance_por_borde = direccion_movimiento * idx
            punto_silueta = punto_inicial + desvio_profundidad + avance_por_borde
            puntos_silueta.append(punto_silueta)
        
        return np.asarray(puntos_silueta)

    def __recortar_mascara_booleana(self,mascara:BoolArray)->BoolArray:

        y, x = np.where(mascara)

        x_inicial = int(np.min(x))
        x_fin = int(np.max(x)) + 1
        y_inicial = int(np.min(y))
        y_fin = int(np.max(y)) + 1

        return mascara[y_inicial: y_fin, x_inicial:x_fin]

    def __rotar_puntos_alrededor_del_pivote(self, puntos: np.ndarray, pivote: np.ndarray, matriz_rotacion: np.ndarray) -> np.ndarray:
        return (np.asarray(puntos, dtype=np.float64) - pivote) @ matriz_rotacion.T + pivote

    def rotar_geometria(self, angulo_grados: float) -> None:
        # Rotamos los puntos ya conocidos en vez de re-detectarlos desde la mascara
        # rotada: eso dependia de cv2.findContours/minAreaRect y se volvia inestable
        # con la interpolacion de cada rotacion. El pivote es el centro real, que por
        # construccion siempre cae en el centro de self.imagen y no se mueve al rotar
        # sobre si mismo; por eso punto_inicial no se rota, solo se traslada.
        x_inicial, x_final, y_inicial, y_final = self.bounding_box_real()
        pivote = np.array([(x_inicial + x_final) / 2.0, (y_inicial + y_final) / 2.0])

        angulo_rad = np.radians(angulo_grados)
        coseno, seno = np.cos(angulo_rad), np.sin(angulo_rad)
        matriz_rotacion = np.array([[coseno, -seno], [seno, coseno]])

        self.borde_norte = self.__rotar_puntos_alrededor_del_pivote(self.borde_norte, pivote, matriz_rotacion)
        self.borde_sur = self.__rotar_puntos_alrededor_del_pivote(self.borde_sur, pivote, matriz_rotacion)
        self.borde_este = self.__rotar_puntos_alrededor_del_pivote(self.borde_este, pivote, matriz_rotacion)
        self.borde_oeste = self.__rotar_puntos_alrededor_del_pivote(self.borde_oeste, pivote, matriz_rotacion)
        self.puntos_esquinas = self.__rotar_puntos_alrededor_del_pivote(self.puntos_esquinas, pivote, matriz_rotacion)

        # Volvemos a arrancar todo desde (0,0), como al construir la pieza. El pivote no
        # roto (punto_inicial) solo se traslada por la misma cantidad.
        todos_los_puntos = np.vstack([self.borde_norte, self.borde_sur, self.borde_este, self.borde_oeste])
        origen_nuevo = todos_los_puntos.min(axis=0)

        self.borde_norte -= origen_nuevo
        self.borde_sur -= origen_nuevo
        self.borde_este -= origen_nuevo
        self.borde_oeste -= origen_nuevo
        self.puntos_esquinas -= origen_nuevo
        self.punto_inicial = np.asarray(self.punto_inicial, dtype=np.float64) - origen_nuevo

class Pieza:
    def __init__(self, imagen: np.ndarray, descriptor_geometria:DescriptorGeometria, rompecabezas: RompecabezasV2, id: int, padding: int):

        if issubclass(imagen.dtype.type, np.floating):
            self.tipo = float
        elif issubclass(imagen.dtype.type, np.integer):
            self.tipo = int
        else:
            raise ValueError("Tipo No Valido")

        self.rompecabezas = rompecabezas
        self.id = id # Representa su ubicacion en el rompecabezas, por lo que es unica
        self.imagen: np.ndarray = imagen
        self.padding_geometria = padding
        self.padding_actual = 0 # padding negro, no de imagen
        self.angulo = 0
        self.geometria= MascaraMuesca(descriptor_geometria)


    def __rango_valido(self) -> tuple:
        if self.tipo == int:
            return 0, 255
        elif self.tipo == float:
            return 0.0, 1.0
        else:
            raise ValueError("Tipo No Valido")

    def __assert_range(self, imagen:np.ndarray):

        rango_minimo, rango_maximo = self.__rango_valido()

        assert imagen.min() >= rango_minimo and imagen.max() <= rango_maximo, \
            f"La imagen debe tener valores en el rango [{rango_minimo}, {rango_maximo}]"

    def actualizar_imagen(self, nueva_imagen:np.ndarray):
        tipo_base_esperado = np.floating if self.tipo == float else np.integer
        assert issubclass(nueva_imagen.dtype.type, tipo_base_esperado), "La nueva imagen debe ser del mismo tipo para mantener consistencia"
        self.__assert_range(nueva_imagen)
        self.imagen = nueva_imagen

    @property
    def dimensiones_imagen(self)-> tuple[int, int]:
        return self.imagen.shape[1], self.imagen.shape[0]

    def __get_cambio_dimensiones_por_giro(self, coordenada_x, coordenada_y, angulo):

        ancho, altura = self.dimensiones_imagen

        radio = np.sqrt(altura * altura + ancho * ancho)

        angulo_del_punto = np.atan2(coordenada_y, coordenada_x)
        nuevo_angulo = angulo_del_punto + np.radians(angulo)

        nueva_altura = np.abs(radio * np.sin(nuevo_angulo))  
        nuevo_ancho = np.abs(radio * np.cos(nuevo_angulo))  

        cambio_ancho = nuevo_ancho - ancho  
        cambio_altura = nueva_altura - altura

        return cambio_ancho, cambio_altura

    def _get_cambio_dimension_maximo_por_giro(self, angulo):
            
        ancho, alto = self.dimensiones_imagen

        # hago las coordenadas relativas al centro de la pieza
        x_inicio = -ancho // 2
        x_fin = ancho // 2

        y_inicio = -alto // 2 
        y_fin = alto // 2 

        # Consigo el cambio de las dos diagonales principales, que son las que se quedarian afuera si o si
        cambio_ancho_diagonal_1, cambio_altura_diagonal_1 = self.__get_cambio_dimensiones_por_giro(x_inicio, y_fin, angulo)
        cambio_ancho_diagonal_2, cambio_altura_diagonal_2 = self.__get_cambio_dimensiones_por_giro(x_fin, y_fin, angulo)

        mayor_cambio_ancho = np.maximum(cambio_ancho_diagonal_1, cambio_ancho_diagonal_2)
        mayor_cambio_altura = np.maximum(cambio_altura_diagonal_1, cambio_altura_diagonal_2)

        mayor_cambio = np.maximum(mayor_cambio_ancho, mayor_cambio_altura)
        return np.maximum(0, mayor_cambio)

    def __calcular_padding_rotacion(self, angulo):

        cambio_mayor = self._get_cambio_dimension_maximo_por_giro(angulo)
        padding_necesario_para_no_cortar_esquinas = cambio_mayor//2 + 1

        # comparamos con el padding actual, si es que hay, no agregos padding al pedo.
        padding_faltante = int(padding_necesario_para_no_cortar_esquinas - self.padding_actual)
        padding_faltante = np.maximum(0, padding_faltante)

        return padding_faltante

    def rotar(self, angulo: float):

        angulo = angulo % 360

        padding_faltante = self.__calcular_padding_rotacion(angulo)

        imagen_actual = self.imagen
        imagen_rotada, _ = enderezar_pieza(imagen_actual, angulo, padding_faltante)

        # cv2.warpAffine con interpolacion cubica puede pasarse del rango de entrada cerca de bordes filosos
        rango_minimo, rango_maximo = self.__rango_valido()
        imagen_rotada = np.clip(imagen_rotada, rango_minimo, rango_maximo).astype(imagen_actual.dtype)

        self.padding_actual += padding_faltante
        self.angulo = (self.angulo + angulo) % 360

        self.actualizar_imagen(imagen_rotada)
        # Rotamos los bordes/esquinas ya conocidos en vez de re-detectarlos desde una
        # mascara rotada (ver rotar_geometria): no hace falta rotar ninguna mascara aca.
        self.geometria.rotar_geometria(angulo)

    def __calcular_punto_de_referencia_mascara(self):
        x_inicial, x_final, y_inicial, y_final = self.geometria.bounding_box_real()
        centro_real_x = (x_inicial + x_final) / 2
        centro_real_y = (y_inicial + y_final) / 2

        alto_imagen, ancho_imagen = self.imagen.shape[:2]
        x_inicio = round(ancho_imagen / 2 - centro_real_x)
        y_inicio = round(alto_imagen / 2 - centro_real_y)

        return x_inicio, y_inicio

    def devolver_pieza(self) -> np.ndarray:
        if self.tipo == float:
            valor_opacidad = 1.0
        elif self.tipo == int:
            valor_opacidad = 255
        else:
            raise ValueError("Tipo No Valido")

        mascara_geometria = self.geometria.generar_mascara()
        alto_mascara, ancho_mascara = mascara_geometria.shape

        x_inicio, y_inicio = self.__calcular_punto_de_referencia_mascara()

        # El centrado puede desviarse en un par de pixeles por el redondeo e interpolacion
        # acumulados entre rotaciones, asi que recortamos al area realmente disponible.
        alto_imagen, ancho_imagen = self.imagen.shape[:2]
        y_inicio_valido, x_inicio_valido = max(y_inicio, 0), max(x_inicio, 0)
        y_fin_valido = min(y_inicio + alto_mascara, alto_imagen)
        x_fin_valido = min(x_inicio + ancho_mascara, ancho_imagen)

        y_inicio_en_mascara, x_inicio_en_mascara = y_inicio_valido - y_inicio, x_inicio_valido - x_inicio
        y_fin_en_mascara = y_inicio_en_mascara + (y_fin_valido - y_inicio_valido)
        x_fin_en_mascara = x_inicio_en_mascara + (x_fin_valido - x_inicio_valido)

        parche_color = self.imagen[y_inicio_valido:y_fin_valido, x_inicio_valido:x_fin_valido]
        mascara_recortada = mascara_geometria[y_inicio_en_mascara:y_fin_en_mascara, x_inicio_en_mascara:x_fin_en_mascara]
        canal_alpha = np.where(mascara_recortada, valor_opacidad, 0).astype(parche_color.dtype)

        return np.dstack([parche_color, canal_alpha])

    def actualizar_pieza_recortada(self, parche_color: np.ndarray) -> None:
        """Inverso de devolver_pieza: escribe un parche del mismo tamaño y area (sin canal alpha)
        de vuelta en self.imagen, sin tocar el padding de alrededor."""
        mascara_geometria = self.geometria.generar_mascara()
        alto_mascara, ancho_mascara = mascara_geometria.shape

        x_inicio, y_inicio = self.__calcular_punto_de_referencia_mascara()

        alto_imagen, ancho_imagen = self.imagen.shape[:2]
        y_inicio_valido, x_inicio_valido = max(y_inicio, 0), max(x_inicio, 0)
        y_fin_valido = min(y_inicio + alto_mascara, alto_imagen)
        x_fin_valido = min(x_inicio + ancho_mascara, ancho_imagen)

        y_inicio_en_mascara, x_inicio_en_mascara = y_inicio_valido - y_inicio, x_inicio_valido - x_inicio
        y_fin_en_mascara = y_inicio_en_mascara + (y_fin_valido - y_inicio_valido)
        x_fin_en_mascara = x_inicio_en_mascara + (x_fin_valido - x_inicio_valido)

        mascara_recortada = mascara_geometria[y_inicio_en_mascara:y_fin_en_mascara, x_inicio_en_mascara:x_fin_en_mascara]
        parche_recortado = parche_color[y_inicio_en_mascara:y_fin_en_mascara, x_inicio_en_mascara:x_fin_en_mascara]

        nueva_imagen = self.imagen.copy()
        destino = nueva_imagen[y_inicio_valido:y_fin_valido, x_inicio_valido:x_fin_valido]
        nueva_imagen[y_inicio_valido:y_fin_valido, x_inicio_valido:x_fin_valido] = np.where(mascara_recortada[..., None], parche_recortado, destino)

        self.actualizar_imagen(nueva_imagen)

    def calcular_borde_color(self, borde:str) -> np.ndarray:
        x_inicio, y_inicio = self.__calcular_punto_de_referencia_mascara()

        punto_de_referencia = np.asarray([x_inicio, y_inicio])

        if borde == "NORTE":
            puntos_en_borde = self.geometria.borde_norte
        elif borde == "SUR":
            puntos_en_borde = self.geometria.borde_sur
        elif borde == "OESTE":
            puntos_en_borde = self.geometria.borde_oeste
        elif borde == "ESTE":
            puntos_en_borde = self.geometria.borde_este
        else:
            raise ValueError("Borde No Valido")
        
        alto_imagen, ancho_imagen = self.imagen.shape[:2]

        borde_de_colores = []
        for punto in puntos_en_borde:
            punto_en_imange = np.round(punto + punto_de_referencia).astype(int)
            # El redondeo e interpolacion acumulados entre rotaciones pueden desviar el punto
            # en un par de pixeles; lo sujetamos al borde de la imagen en vez de fallar.
            x_punto = np.clip(punto_en_imange[0], 0, ancho_imagen - 1)
            y_punto = np.clip(punto_en_imange[1], 0, alto_imagen - 1)
            borde_de_colores.append(self.imagen[y_punto, x_punto])

        return np.asarray(borde_de_colores)