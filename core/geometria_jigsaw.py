"""
Módulo de Geometría de Rompecabezas con Encastres (Tabs & Blanks / Saliente y Entrante).
Genera curvas bezier y analíticas continuas entre fichas adyacentes tipo Jigsaw.
"""

from typing import Dict, List, Tuple, Any, Optional
from core.tipos import BoolArray
import numpy as np
from dataclasses import dataclass
import cv2


MINIMO_CONTENIDO_MASCARA_FLOAT = 0.15
MINIMO_CONTENIDO_MASCARA_INT = 40

#Tabla finita de generacion de bordes
TABLA_BORDES_DISCRETOS = {
    "perfiles": ["gaussian", "semicircular"],
    "posiciones": (0.35, 0.425, 0.50, 0.575, 0.65),
    "profundidades": (0.16, 0.20, 0.24),
    "anchos": (0.24, 0.28, 0.32, 0.36),            
    "ancho": 0.30,
}

__all__ = ["JigsawGridGeometry", "TABLA_BORDES_DISCRETOS"]

@dataclass
class DescriptorGeometria:
    borde_norte: np.ndarray
    borde_sur: np.ndarray
    borde_este: np.ndarray
    borde_oeste: np.ndarray
    punto_inicial: np.ndarray
    dimensiones_base: tuple[int, int]


class JigsawGridGeometry:
    """
    Gestiona la coherencia global de los encastres de una grilla R x C.
    Garantiza que si el borde Este de la pieza (r, c) es Saliente (+1),
    el borde Oeste de la pieza (r, c+1) sea el encastre Entrante (-1) complementario exacto.
    """

    def __init__(
        self,
        filas: int,
        columnas: int,
        altura_imagen: int,
        ancho_imagen: int,
        seed: Optional[int] = 42,
        profile_types: Optional[List[str]] = ["gaussian", "semicircular", "ancha"],
        discrete: bool = False,
    ):
        self.filas = filas
        self.columnas = columnas
        self.altura_imagen = altura_imagen
        self.ancho_imagen = ancho_imagen
 
        self.altura_pieza = altura_imagen // filas
        self.ancho_pieza = ancho_imagen // columnas
        self.utilizar_muecas_pre_definidas = discrete
 
        rng = np.random.default_rng(seed)

        # Si es -1, la pieza del indice es la 'hembra' en la muesca del lado derecho
        # Si es 1, la pieza del indice es la 'macho' en la muesca del lado derecho
        self.direccion_muesca_horizontal = rng.choice([1, -1], size=(filas, columnas - 1))

        # Si es -1, la pieza del indice es la 'hembra' en la muesca del lado inferior
        # Si es 1, la pieza del indice es la 'macho' en la muesca del lado inferior
        self.direccion_muesca_vertical = rng.choice([1, -1], size=(filas - 1, columnas))
 
        if self.utilizar_muecas_pre_definidas:
            self.perfiles_permitidos = TABLA_BORDES_DISCRETOS["perfiles"]
            posibles_ubicacion_centros = TABLA_BORDES_DISCRETOS["posiciones"]
            posibles_profundidad = TABLA_BORDES_DISCRETOS["profundidades"]
            posibles_anchos = TABLA_BORDES_DISCRETOS["anchos"]

        else:
            # Los valores son porcentajes sobre el largo del borde
            self.perfiles_permitidos = profile_types
            posibles_ubicacion_centros = np.linspace(0.40, 0.60, 21)
            posibles_profundidad = np.linspace(0.18, 0.22, 5)
            posibles_anchos = np.linspace(0.28, 0.34, 7)

        self.tipos_perfiles_horizontales, self.tipos_perfiles_verticales = self._elegir_valor_para_border_horizontales_verticales(self.perfiles_permitidos, rng)
        self.centros_horizontales, self.centros_verticales = self._elegir_valor_para_border_horizontales_verticales(posibles_ubicacion_centros, rng)
        self.profundidades_horizontales, self.profundidades_verticales = self._elegir_valor_para_border_horizontales_verticales(posibles_profundidad, rng)
        self.anchos_horizontales, self.anchos_verticales = self._elegir_valor_para_border_horizontales_verticales(posibles_anchos, rng)
 
        self._siluetas_verticales = {} 
        self._siluetas_horizontales = {}
        self._generar_puntos_silueta_por_pieza()

# region Elegir por Borde
    def _elegir_valor_para_borde_horizontal_por_pieza(self, lista_de_valores, generador):
        return generador.choice(lista_de_valores, size=(self.filas, self.columnas - 1))

    def _elegir_valor_para_borde_vertical_por_pieza(self, lista_de_valores, generador):
        return generador.choice(lista_de_valores, size=(self.filas-1, self.columnas))

    def _elegir_valor_para_border_horizontales_verticales(self, lista_de_valores, generador):
        valores_borde_horizontales = self._elegir_valor_para_borde_horizontal_por_pieza(lista_de_valores, generador)
        valores_borde_vertical = self._elegir_valor_para_borde_vertical_por_pieza(lista_de_valores, generador)
        return valores_borde_horizontales, valores_borde_vertical
# endregion

# region Generacion de Siluetas
    def _generar_silueta_semicircular(self, distacia_al_centro_de_la_muesca: float) -> float:
        # Genera un semi-circulo unitario para -1 <= x <= 1 y 0 en el resto
        
        # (X^2 + Y^2 = 1) -> (X^2 - 1 = Y^2)
        altura_del_punto = 1.0 - (distacia_al_centro_de_la_muesca ** 2)

        # Clipeamos la parte liza del borde
        altura_del_punto = max(0.0, altura_del_punto)
        altura_del_punto = np.sqrt(altura_del_punto) 
        return float(altura_del_punto)

    def _generar_silueta_unitaria_copa_sinuisal(self, distacia_al_centro_de_la_muesca: float, coeficiente_puntiagudo_copa) -> float:

        # Coseno con copa superior entre -1 y 1, despues nula
        if np.abs(distacia_al_centro_de_la_muesca) <= 1:
            silueta_unitaria = np.cos(distacia_al_centro_de_la_muesca * np.pi / 2.0)
        else:
            silueta_unitaria = 0

        return float(silueta_unitaria ** coeficiente_puntiagudo_copa)

    def _generar_silueta_ancha(self, distacia_al_centro_de_la_muesca):
        return self._generar_silueta_unitaria_copa_sinuisal(distacia_al_centro_de_la_muesca, 1.3)

    def _generar_siluesta_gaussiana(self, distacia_al_centro_de_la_muesca):
        distancia_normalizada = distacia_al_centro_de_la_muesca ** 2
        if distancia_normalizada <= 1:
            # Utilizamos una constante para que 1 este muy cerca de 0. Con 7 -> 0.001
            silueta_gaussiana = np.exp(-7 * distancia_normalizada)
        else:
            silueta_gaussiana = 0
        return float(silueta_gaussiana)

    def _generar_siluesta_borde(self):
        return float(0)

    def _generar_silueta_por_tipo(self, prof: str, distacia_al_centro_de_la_muesca: float) -> float:
        #Desplazamiento perpendicular normalizado segun el perfil del encastre

        if prof == "semicircular":
            return self._generar_silueta_semicircular(distacia_al_centro_de_la_muesca)
        elif prof == "ancha":
            return self._generar_silueta_ancha(distacia_al_centro_de_la_muesca)
        elif prof == "gaussian":
            return self._generar_siluesta_gaussiana(distacia_al_centro_de_la_muesca)
        elif prof == "borde":
            return self._generar_siluesta_borde()
        else:
            raise ValueError("Tipo No Valido")

    def generar_puntos_silueta_en_interseccion(self, inicio_interseccion, fila, columna, largo_borde, es_vertical):

        if es_vertical:
            perfil_entre_piezas = self.tipos_perfiles_verticales[fila, columna]
            direccion_muestra = self.direccion_muesca_vertical[fila, columna]
            profundidad_muesca = self.profundidades_verticales[fila, columna] * self.altura_pieza
            ancho_muesca = self.anchos_verticales[fila, columna]
            centro_muesca = self.centros_verticales[fila, columna]
        else:
            perfil_entre_piezas = self.tipos_perfiles_horizontales[fila, columna]
            direccion_muestra = self.direccion_muesca_horizontal[fila, columna]
            profundidad_muesca = self.profundidades_horizontales[fila, columna] * self.ancho_pieza
            ancho_muesca = self.anchos_horizontales[fila, columna]
            centro_muesca = self.centros_horizontales[fila, columna]

        radio_muesca = ancho_muesca / 2

        puntos_muesca = []
        for pixel_en_borde in range(0, largo_borde):
            progreso_en_borde = pixel_en_borde / (largo_borde - 1)
            distancia_al_centro = progreso_en_borde - centro_muesca
            distancia_al_centro_relativa = distancia_al_centro / radio_muesca

            altura_perfil_relativa = self._generar_silueta_por_tipo(perfil_entre_piezas, distancia_al_centro_relativa)
            altura_perfil = altura_perfil_relativa * profundidad_muesca
            altura_perfil = altura_perfil * direccion_muestra

            if es_vertical:
                punto_muesca = [pixel_en_borde, altura_perfil]
            else:
                punto_muesca = [altura_perfil, pixel_en_borde]

            punto_muesca = np.array(punto_muesca, dtype=np.float32)
            puntos_muesca.append(inicio_interseccion + punto_muesca) 

        return puntos_muesca
    
    def _generar_puntos_silueta_por_pieza(self) -> None:
        """Calcula una única vez cada curva de unión interior compartida."""

        # 1. Costuras horizontales: entre fila r y r+1
        for fila in range(self.filas - 1):
            for columna in range(self.columnas):
                
                coordena_y_interseccion = (fila + 1) * self.altura_pieza
                coordenada_x_inicio_interseccion = columna * self.ancho_pieza 
                
                inicio_interseccion = np.array([coordenada_x_inicio_interseccion, coordena_y_interseccion], dtype=np.float32)

                puntos_muesca = self.generar_puntos_silueta_en_interseccion(inicio_interseccion, fila, columna, largo_borde=self.ancho_pieza, es_vertical=True)   
                self._siluetas_verticales[(fila, columna)] = np.array(puntos_muesca, dtype=np.float32)
 
        # 2. Costuras verticales: entre columna c y c+1
        for fila in range(self.filas):
            for columna in range(self.columnas - 1):

                coordena_x_interseccion = (columna + 1) * self.ancho_pieza
                coordenada_y_inicio_interseccion = fila * self.altura_pieza 

                inicio_interseccion = np.array([coordena_x_interseccion, coordenada_y_inicio_interseccion], dtype=np.float32)

                puntos_muesca = self.generar_puntos_silueta_en_interseccion(inicio_interseccion, fila, columna, largo_borde=self.altura_pieza, es_vertical=False)   
                self._siluetas_horizontales[(fila, columna)] = np.array(puntos_muesca, dtype=np.float32)

# endregion

# region Devolver Tipo de Borde

    def _tipo_de_borde_por_direccion(self, direccion:int) -> str:
        if direccion == 1:
            return "SALIENTE"
        elif direccion == -1:
            return "ENTRANTE"
        else:
            raise ValueError("Direccion No Valida")

    def _encontrar_tipo_de_borde_norte(self, fila:int, columna:int) -> str:
        if fila == 0:
            return "PLANO"

        direccion_muesca_pieza_superior = self.direccion_muesca_vertical[fila - 1, columna]

        # Invierto para tener mi direccion
        return self._tipo_de_borde_por_direccion(-direccion_muesca_pieza_superior)

    def _encontrar_tipo_de_borde_sur(self, fila:int, columna:int) -> str:
        if fila == (self.filas - 1):
            return "PLANO"

        direccion_muesca_pieza = self.direccion_muesca_vertical[fila, columna]
        return self._tipo_de_borde_por_direccion(direccion_muesca_pieza)

    def _encontrar_tipo_de_borde_oeste(self, fila:int, columna:int) -> str:
        if columna == 0:
            return "PLANO"

        # Invierto para tener mi direccion
        direccion_muesca_pieza_izquierda = self.direccion_muesca_horizontal[fila, columna - 1]
        return self._tipo_de_borde_por_direccion(-direccion_muesca_pieza_izquierda)

    def _encontrar_tipo_de_borde_este(self, fila:int, columna:int) -> str:
        if columna == (self.columnas - 1):
            return "PLANO"

        direccion_muesca_pieza = self.direccion_muesca_horizontal[fila, columna]
        return self._tipo_de_borde_por_direccion(direccion_muesca_pieza)

    def get_piece_edge_types(self, fila: int, columna: int) -> Dict[str, str]:
        """Devuelve 'PLANO', 'SALIENTE', 'ENTRANTE' para cada lado N, S, E, W de la pieza (fila, columna)."""
        tipo_borde_norte = self._encontrar_tipo_de_borde_norte(fila, columna)
        tipo_borde_sur = self._encontrar_tipo_de_borde_sur(fila, columna)
        tipo_borde_oeste = self._encontrar_tipo_de_borde_oeste(fila,columna)
        tipo_borde_este = self._encontrar_tipo_de_borde_este(fila,columna)
        return {"NORTE": tipo_borde_norte, "SUR": tipo_borde_sur, "OESTE": tipo_borde_oeste, "ESTE": tipo_borde_este}

# endregion

# region Devolver Tipo Perfil Borde
    def _encontrar_perfil_de_borde_norte(self, fila:int, columna:int) -> str:
        if fila == 0:
            return "none"
        
        # Agarro el de la pieza de arriba
        return self.tipos_perfiles_verticales[fila - 1, columna]


    def _encontrar_perfil_de_borde_sur(self, fila:int, columna:int) -> str:
        if fila == (self.filas - 1):
            return "none"
        return self.tipos_perfiles_verticales[fila, columna]

    def _encontrar_perfil_de_borde_oeste(self, fila:int, columna:int) -> str:
        if columna == 0:
            return "none"

        # Agarro el de la pieza a mi izquierda
        return self.tipos_perfiles_horizontales[fila, columna - 1]

    def _encontrar_perfil_de_borde_este(self, fila:int, columna:int) -> str:
        if columna == (self.columnas - 1):
            return "none"

        return self.tipos_perfiles_horizontales[fila, columna]

    def get_piece_edge_curves(self, fila: int, columna: int) -> Dict[str, str]:
        """Devuelve el perfil ('none', 'standard', 'circular', 'wide') para cada lado."""
        tipo_perfil_borde_norte = self._encontrar_perfil_de_borde_norte(fila,columna)
        tipo_perfil_borde_sur = self._encontrar_perfil_de_borde_sur(fila,columna)
        tipo_perfil_borde_oeste = self._encontrar_perfil_de_borde_oeste(fila,columna)
        tipo_perfil_borde_este = self._encontrar_perfil_de_borde_este(fila,columna)
        return {"NORTE": tipo_perfil_borde_norte, "SUR": tipo_perfil_borde_sur, "OESTE": tipo_perfil_borde_oeste, "ESTE": tipo_perfil_borde_este}

# endregion

# region Devolver Silueta Borde

    def _generar_borde_horizontal_plano(self, columna:int, y_fijo) -> np.ndarray:
        x_inicio = columna * self.ancho_pieza
        x_final = x_inicio + self.ancho_pieza
        return np.array([[x, y_fijo] for x in range(x_inicio, x_final)])

    def _generar_borde_vertical_plano(self, fila:int, x_fijo) -> np.ndarray:
        y_inicio = fila * self.altura_pieza
        y_final = y_inicio + self.altura_pieza
        return np.array([[x_fijo, y] for y in range(y_inicio, y_final)])

    def _encontrar_silueta_borde_norte(self, fila: int, columna: int) -> np.ndarray:
        if fila == 0:
            return self._generar_borde_horizontal_plano(columna, 0)
        else:
            # Voy a la pieza arriba para encontrarla
            return self._siluetas_verticales[(fila - 1, columna)].copy()

    def _encontrar_silueta_borde_sur(self, fila: int, columna: int) -> np.ndarray:
        if fila == self.filas - 1:

            return self._generar_borde_horizontal_plano(columna, self.altura_imagen - 1)
        else:
            return self._siluetas_verticales[(fila, columna)].copy()

    def _encontrar_silueta_borde_oeste(self, fila: int, columna: int) -> np.ndarray:
        if columna == 0:
            return self._generar_borde_vertical_plano(fila, 0)
        else:
            # Voy a la pieza izquierda para encontrarla
            return self._siluetas_horizontales[(fila, columna - 1)].copy()

    def _encontrar_silueta_borde_este(self, fila: int, columna: int) -> np.ndarray:
        if columna == self.columnas - 1:
            return self._generar_borde_vertical_plano(fila, self.ancho_imagen - 1)
        else:
            return self._siluetas_horizontales[(fila, columna)].copy()

    def get_piece_edges(self, fila: int, columna: int) -> Dict[str, np.ndarray]:
        """Retorna las 4 curvas del contorno de la pieza (r, c) orientadas en sentido horario."""

        silueta_norte = self._encontrar_silueta_borde_norte(fila,columna)
        silueta_sur = self._encontrar_silueta_borde_sur(fila,columna)
        silueta_oeste = self._encontrar_silueta_borde_oeste(fila,columna)
        silueta_este = self._encontrar_silueta_borde_este(fila,columna)

        return {"NORTE": silueta_norte, "ESTE": silueta_este, "SUR": silueta_sur, "OESTE": silueta_oeste}

# endregion

    def get_piece_polygon(self, fila: int, columna: int) -> np.ndarray:
        """Retorna el polígono 2D cerrado en sentido horario."""
        edges = self.get_piece_edges(fila, columna)
        # Doy vuelta sur y oeste para que empiezen de deracha a izquierda y de abajo hacia arriba
        return np.vstack([edges["NORTE"], edges["ESTE"], np.flip(edges["SUR"], axis=0), np.flip(edges["OESTE"], axis=0)])

    def _bounding_box_poligono(self, poligono_silueta: np.ndarray):
        x_inicio_silueta = int(np.min(poligono_silueta[:, 0]))
        x_fin_silueta = int(np.max(poligono_silueta[:, 0])) + 1

        y_inicio_silueta = int(np.min(poligono_silueta[:, 1]))
        y_fin_silueta = int(np.max(poligono_silueta[:, 1])) + 1

        return x_inicio_silueta, x_fin_silueta, y_inicio_silueta, y_fin_silueta

    def _dimensiones_poligono(self, poligono_silueta: np.ndarray):
        x_inicio_silueta, x_fin_silueta, y_inicio_silueta, y_fin_silueta = self._bounding_box_poligono(poligono_silueta)
        ancho_silueta = x_fin_silueta - x_inicio_silueta
        alto_silueta = y_fin_silueta - y_inicio_silueta

        return ancho_silueta, alto_silueta


    def extract_piece_image(
        self,
        full_image: np.ndarray,
        fila: int,
        columna: int,
        padding: int = 35,
        anular_pixeles_fuera_mascara = True
    ) -> Tuple[np.ndarray, BoolArray, Tuple[int, int]]:
        """
        Recorta la pieza correspondiente a (r, c) con su forma real de encastre sobre fondo negro (0, 0, 0).
        Soporta arrays float64 [0, 1] y uint8.

        Returns:
            - piece_img: Imagen recortada centrada sobre fondo negro.
            - piece_mask: Máscara binaria (uint8: 255 pieza, 0 fondo).
            - (offset_x, offset_y): Coordenadas globales de la esquina sup-izq del parche.
        """
        poligono_silueta_pieza = self.get_piece_polygon(fila, columna)
        poligono_silueta_pieza = np.round(poligono_silueta_pieza).astype(np.int32)

        x_inicio_silueta, x_fin_silueta, y_inicio_silueta, y_fin_silueta = self._bounding_box_poligono(poligono_silueta_pieza)

        ancho_silueta, alto_silueta = self._dimensiones_poligono(poligono_silueta_pieza)

        # Lo hago siempre par para evitar problemas
        if padding % 2 != 0:
            padding += 1

        ancho_silueta_padded = ancho_silueta + padding
        alto_silueta_padded = alto_silueta + padding

        canales = full_image.shape[2] if full_image.ndim == 3 else 1
        pieza_recortada = np.zeros((alto_silueta_padded, ancho_silueta_padded, canales), dtype=full_image.dtype)

        mascara = np.zeros((alto_silueta_padded, ancho_silueta_padded), dtype=np.uint8)
        
        offset = padding // 2
        x_inicio_imagen_destino = int(offset)
        x_fin_imagen_destino = x_inicio_imagen_destino + ancho_silueta
        y_inicio_imagen_destino = int(offset)
        y_fin_imagen_destino = y_inicio_imagen_destino + alto_silueta

        if not (x_fin_silueta > x_inicio_silueta) or not (y_fin_silueta > y_inicio_silueta):
            raise ValueError("Rangos Invalidos")

        patch = full_image[y_inicio_silueta:y_fin_silueta, x_inicio_silueta:x_fin_silueta]
        pieza_recortada[y_inicio_imagen_destino:y_fin_imagen_destino, x_inicio_imagen_destino:x_fin_imagen_destino, :] = patch

        # Asegurar valor mínimo pequeño para no confundir con fondo negro puro
        if issubclass(full_image.dtype.type, np.floating):
            pieza_recortada = np.maximum(pieza_recortada, MINIMO_CONTENIDO_MASCARA_FLOAT)
        else:
            pieza_recortada = np.maximum(pieza_recortada, MINIMO_CONTENIDO_MASCARA_INT)

        #mando la mascara para que el inicio coincida con cero y despues offseteo por el padding
        poligono_mascara = poligono_silueta_pieza.copy()
        poligono_mascara[:, 0] -= x_inicio_silueta
        poligono_mascara[:, 1] -= y_inicio_silueta
        poligono_mascara[:, 0] += offset
        poligono_mascara[:, 1] += offset

        cv2.fillPoly(mascara, [poligono_mascara], 255)
        mascara = mascara == 255

        if anular_pixeles_fuera_mascara:
            for canal in range(canales):
                pieza_recortada[:, :, canal] = np.where(mascara, pieza_recortada[:, :, canal], 0)

        return pieza_recortada, mascara, (x_inicio_silueta, y_inicio_silueta)

    def extraer_geometria_por_pieza(
        self,
        fila: int,
        columna: int,
    ) -> DescriptorGeometria:

        muescas_por_borde = self.get_piece_edges(fila, columna)
        coordenada_x_esquina = columna * self.ancho_pieza
        coordenada_y_esquina = fila * self.altura_pieza

        descriptor_geometria = DescriptorGeometria(
            muescas_por_borde["NORTE"],
            muescas_por_borde["SUR"],
            muescas_por_borde["ESTE"],
            muescas_por_borde["OESTE"],
            np.asarray([coordenada_x_esquina, coordenada_y_esquina]),
            (self.ancho_pieza, self.altura_pieza)
        )

        return descriptor_geometria

