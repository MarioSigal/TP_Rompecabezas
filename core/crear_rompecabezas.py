"""

"""

from typing import Callable, Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import numpy as np
import cv2

try:
    from .preparacion_imagenes import asegurar_rgb_float
    from .degradaciones import (
    agregar_ruido_gaussiano,
    agregar_ruido_sal_y_pimienta,
    agregar_ruido_uniforme,
    agregar_ruido_rayleigh,
    componer_degradaciones,
    DegradacionPorPiezaNivel2,
    sortear_variante_nivel2,
    TramaMixtaPorPieza,
    DegradacionPorPiezaRotacion
    )
    from .geometria_jigsaw import JigsawGridGeometry
    from .analizador_rotacion import rotar_imagen_ortogonal, enderezar_pieza
except ImportError:
    from preparacion_imagenes import asegurar_rgb_float
    from degradaciones import (
    agregar_ruido_gaussiano,
    agregar_ruido_sal_y_pimienta,
    agregar_ruido_uniforme,
    agregar_ruido_rayleigh,
    componer_degradaciones,
    DegradacionPorPiezaNivel2,
    sortear_variante_nivel2,
    TramaMixtaPorPieza,
    )
    from geometria_jigsaw import JigsawGridGeometry
    from analizador_rotacion import aplicar_filtro_rayas_horizontales, rotar_imagen_ortogonal, enderezar_pieza

__all__ = [
    "Rompecabezas",
    "garantizar_dimensiones_para_divisibilidad",
    "cortar_imagen_en_piezas",
    "barajar_piezas",
    "pegar_piezas",
    "armar_caso_rompecabezas",
    "crear_rompecabezas_nivel",
    "crear_dataset_desafio_30",
    "pre_proceso_imagen_para_rompecabezas"
]

class Rompecabezas:

    def __init__(
        self,
        piezas: List[np.ndarray],
        cantidad_filas: int,
        cantidad_columnas: int,
        posicion_real: Dict[int, Tuple[int, int]],
        rotacion_real: Optional[Dict[int, float]] = None,
        tiene_ranuras: bool = False,
        imagen_base: Optional[np.ndarray] = None,
        imagen_degradada: Optional[np.ndarray] = None,
        metadatos: Optional[Dict[str, Any]] = None,
    ):
        self.piezas = piezas
        self.cantidad_filas = cantidad_filas
        self.cantidad_columnas = cantidad_columnas
        self.posicion_real = posicion_real
        self.rotacion_real = rotacion_real or {p: 0.0 for p in range(len(piezas))}
        self.tiene_ranuras = tiene_ranuras
        self.imagen_base = imagen_base
        self.imagen_degradada = imagen_degradada
        self.metadatos = metadatos or {}

    @property
    def cantidad_piezas(self) -> int:
        return len(self.piezas)

    @property
    def alto_pieza_px(self) -> int:
        return self.piezas[0].shape[0]

    @property
    def ancho_pieza_px(self) -> int:
        return self.piezas[0].shape[1]

    def obtener_matriz_correcta(self) -> np.ndarray:
        """Matriz 2D solución donde grilla[r, c] = id_pieza."""
        matriz = np.full((self.cantidad_filas, self.cantidad_columnas), fill_value=-1, dtype=int)
        for id_pieza, (fila, col) in self.posicion_real.items():
            matriz[fila, col] = id_pieza
        return matriz

    def obtener_adyacencias_correctas(self) -> Dict[str, Dict[int, int]]:
        """
        Retorna las adyacencias verdaderas:
        - 'horizontal': {id_izq: id_der}
        - 'vertical': {id_arriba: id_abajo}
        """
        grilla = self.obtener_matriz_correcta()
        return {
            "horizontal": {int(grilla[r, c]): int(grilla[r, c + 1])
                           for r in range(self.cantidad_filas)
                           for c in range(self.cantidad_columnas - 1)},
            "vertical": {int(grilla[r, c]): int(grilla[r + 1, c])
                         for r in range(self.cantidad_filas - 1)
                         for c in range(self.cantidad_columnas)},
        }

    def pegar_piezas(
        self,
        grilla_propuesta: Optional[np.ndarray] = None,
        piezas: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
       
        if grilla_propuesta is None:
            grilla_propuesta = self.obtener_matriz_correcta()

        lista_piezas = piezas if piezas is not None else self.piezas



        if self.imagen_base is not None:
            h, w = self.imagen_base.shape[:2]
            tile_h = h // self.cantidad_filas
            tile_w = w // self.cantidad_columnas
        else:
            tile_h = lista_piezas[0].shape[0]
            tile_w = lista_piezas[0].shape[1]
            h = self.cantidad_filas * tile_h
            w = self.cantidad_columnas * tile_w

        canales = lista_piezas[0].shape[2] if lista_piezas[0].ndim == 3 else 1
        lienzo = np.zeros((h, w, canales), dtype=np.float64)
        grilla_arr = np.asarray(grilla_propuesta)

        es_forma = (self.tiene_ranuras) or any(
            p.shape[0] > tile_h or p.shape[1] > tile_w for p in lista_piezas[:min(len(lista_piezas), 3)]
        )

        if not es_forma:
            for r in range(self.cantidad_filas):
                for c in range(self.cantidad_columnas):
                    id_p = int(grilla_arr[r, c])
                    if 0 <= id_p < len(lista_piezas):
                        p_img = lista_piezas[id_p]
                        ph, pw = p_img.shape[:2]
                        if ph == tile_h and pw == tile_w:
                            lienzo[r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w] = p_img
                        else:
                            patch_res = cv2.resize(p_img, (tile_w, tile_h))
                            if patch_res.ndim == 2 and canales == 3:
                                patch_res = np.stack([patch_res] * 3, axis=-1)
                            lienzo[r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w] = patch_res
        else:
            for r in range(self.cantidad_filas):
                for c in range(self.cantidad_columnas):
                    id_p = int(grilla_arr[r, c])
                    if 0 <= id_p < len(lista_piezas):
                        p_img = lista_piezas[id_p]
                        ph, pw = p_img.shape[:2]
                        pad_y = max(0, (ph - tile_h) // 2)
                        pad_x = max(0, (pw - tile_w) // 2)

                        y0_dst = r * tile_h - pad_y
                        x0_dst = c * tile_w - pad_x
                        y1_dst = y0_dst + ph
                        x1_dst = x0_dst + pw

                        y0_src = max(0, -y0_dst)
                        x0_src = max(0, -x0_dst)
                        y1_src = ph - max(0, y1_dst - h)
                        x1_src = pw - max(0, x1_dst - w)

                        y0_dst = max(0, y0_dst)
                        x0_dst = max(0, x0_dst)
                        y1_dst = min(h, y1_dst)
                        x1_dst = min(w, x1_dst)

                        if y1_src > y0_src and x1_src > x0_src and y1_dst > y0_dst and x1_dst > x0_dst:
                            patch_src = p_img[y0_src:y1_src, x0_src:x1_src]
                            mask = np.any(patch_src > 0.005, axis=-1) if patch_src.ndim == 3 else (patch_src > 0.005)

                            for ch in range(canales):
                                cur_ch = lienzo[y0_dst:y1_dst, x0_dst:x1_dst, ch]
                                src_ch = patch_src[:, :, ch] if patch_src.ndim == 3 else patch_src
                                lienzo[y0_dst:y1_dst, x0_dst:x1_dst, ch] = np.where(mask, src_ch, cur_ch)

            
            mascara_negra = np.all(lienzo < 0.005, axis=-1) if canales == 3 else (lienzo < 0.005)
            pct_negro = float(np.mean(mascara_negra))
            if 0 < pct_negro < 0.20:
                mask_inpaint = mascara_negra.astype(np.uint8) * 255
                lienzo_u8 = np.clip(lienzo * 255.0, 0, 255).astype(np.uint8)
                inpa_u8 = cv2.inpaint(lienzo_u8, mask_inpaint, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
                
                mask_rem = ((np.all(inpa_u8 == 0, axis=-1) if canales == 3 else (inpa_u8 == 0)) & (mask_inpaint > 0)).astype(np.uint8) * 255
                if np.any(mask_rem > 0):
                    inpa_u8 = cv2.inpaint(inpa_u8, mask_rem, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
                lienzo = inpa_u8.astype(np.float64) / 255.0

        if canales == 1:
            lienzo = lienzo.squeeze(axis=-1)
        return np.clip(lienzo, 0.0, 1.0)



def garantizar_dimensiones_para_divisibilidad(
    imagen: np.ndarray,
    cantidad_filas: int,
    cantidad_columnas: int,
) -> np.ndarray:
    alto, ancho = imagen.shape[:2]
    alto_ajustado = (alto // cantidad_filas) * cantidad_filas
    ancho_ajustado = (ancho // cantidad_columnas) * cantidad_columnas
    return imagen[:alto_ajustado, :ancho_ajustado].copy()


def cortar_imagen_en_piezas(
    imagen: np.ndarray,
    cantidad_filas: int,
    cantidad_columnas: int,
) -> List[np.ndarray]:
    alto, ancho = imagen.shape[:2]
    alto_pieza = alto // cantidad_filas
    ancho_pieza = ancho // cantidad_columnas

    piezas = []
    for r in range(cantidad_filas):
        for c in range(cantidad_columnas):
            p = imagen[r * alto_pieza : (r + 1) * alto_pieza, c * ancho_pieza : (c + 1) * ancho_pieza].copy()
            piezas.append(p)
    return piezas


def barajar_piezas(
    piezas_ordenadas: List[np.ndarray],
    cantidad_columnas: int,
    generador: np.random.Generator,
) -> Tuple[List[np.ndarray], Dict[int, Tuple[int, int]]]:
    total_piezas = len(piezas_ordenadas)
    permutacion = generador.permutation(total_piezas)
    piezas_barajadas = [piezas_ordenadas[i] for i in permutacion]

    posicion_real = {}
    for id_nuevo, id_orig in enumerate(permutacion):
        posicion_real[id_nuevo] = (int(id_orig) // cantidad_columnas, int(id_orig) % cantidad_columnas)

    return piezas_barajadas, posicion_real


def pegar_piezas(piezas: List[np.ndarray], grilla: np.ndarray) -> np.ndarray:
    grilla_arr = np.asarray(grilla)
    filas, columnas = grilla_arr.shape
    alto_p, ancho_p = piezas[0].shape[:2]
    canales = piezas[0].shape[2] if piezas[0].ndim == 3 else 1

    # Detectar si son piezas Jigsaw (con fondo negro o dimensiones variables con padding)
    es_jigsaw = any(p.shape[:2] != (alto_p, ancho_p) for p in piezas) or any(
        np.mean(np.all(p < 0.005, axis=-1) if p.ndim == 3 else (p < 0.005)) > 0.05
        for p in piezas[:min(3, len(piezas))]
    )
    if es_jigsaw:
        puz = Rompecabezas(
            piezas=piezas,
            cantidad_filas=filas,
            cantidad_columnas=columnas,
            posicion_real={},
            tiene_ranuras=True,
            imagen_base=None,
            imagen_degradada=None,
        )
        return puz.pegar_piezas(grilla_arr)

    lienzo = np.zeros((filas * alto_p, columnas * ancho_p, canales), dtype=np.float64)
    for r in range(filas):
        for c in range(columnas):
            id_p = int(grilla_arr[r, c])
            if 0 <= id_p < len(piezas):
                patch = piezas[id_p]
                if patch.ndim == 2 and canales == 3:
                    patch = np.stack([patch] * 3, axis=-1)
                lienzo[r * alto_p : (r + 1) * alto_p, c * ancho_p : (c + 1) * ancho_p] = patch

    return np.clip(lienzo.squeeze(), 0.0, 1.0)

def cortar_ranuras_en_piezas(imagen:np.ndarray, jigsaw:JigsawGridGeometry, padding = 30):
    piezas_ordenadas = []
    piezas_mascaras = []
    for r in range(jigsaw.filas):
        for c in range(jigsaw.columnas):
            pieza_img, mascara, _ = jigsaw.extract_piece_image(imagen, r, c, padding=padding)
            # la funcion devuelve mascara numerica, pero queremos una mascara booleana
            mascara = mascara == 255
            piezas_ordenadas.append(pieza_img)
            piezas_mascaras.append(mascara)

    return piezas_ordenadas, piezas_mascaras

def armar_caso_rompecabezas(
    imagen_base: np.ndarray,
    cantidad_filas: int,
    cantidad_columnas: int,
    degradacion_global: Optional[Callable[[np.ndarray, np.random.Generator], np.ndarray]] = None,
    degradacion_por_pieza: Optional[Callable[[np.ndarray, int, np.random.Generator], np.ndarray]] = None,
    semilla: int = 42,
    tiene_ranuras: bool = False,
    metadatos_adicionales: Optional[Dict[str, Any]] = None,
    barajar: Optional[bool] = True
) -> Rompecabezas:
    """Generador base para rompecabezas con hooks de degradación."""

    rng = np.random.default_rng(semilla)
    img_ajustada = pre_proceso_imagen_para_rompecabezas(imagen_base, cantidad_filas, cantidad_columnas)
    altura, ancho, canales = img_ajustada.shape

    if degradacion_global is not None:
        img_degradada = degradacion_global(img_ajustada, rng)
    else:
        img_degradada = img_ajustada.copy()

    # Cortamos las piezas con o sin ranura
    # Si tienen ranura, ya tiene mascara
    if not tiene_ranuras:
        piezas_ordenadas = cortar_imagen_en_piezas(img_degradada, cantidad_filas, cantidad_columnas)
    else:
        jigsaw = JigsawGridGeometry(cantidad_filas, cantidad_columnas, altura, ancho, seed=semilla, discrete=True)
        piezas_ordenadas, piezas_mascaras = cortar_ranuras_en_piezas(img_degradada, jigsaw)
    
    if degradacion_por_pieza is not None:

        if not isinstance(degradacion_por_pieza, list):
            degradacion_por_pieza = [degradacion_por_pieza]

        for degradacion in degradacion_por_pieza: 
            piezas_degradadas = []
            nuevas_mascaras = []
            for idx, pieza in enumerate(piezas_ordenadas):
                if piezas_mascaras:
                    mascara = piezas_mascaras[idx]
                    pieza_degradada, nueva_mascara = degradacion(pieza, idx, rng, mascara)
                else:
                    pieza_degradada, nueva_mascara = degradacion(pieza, idx, rng)

                piezas_degradadas.append(pieza_degradada)
                nuevas_mascaras.append(nueva_mascara)

            piezas_ordenadas = piezas_degradadas
            piezas_mascaras = nuevas_mascaras

        grilla_ord = np.arange(cantidad_filas * cantidad_columnas).reshape(cantidad_filas, cantidad_columnas)
        img_degradada = pegar_piezas(piezas_ordenadas, grilla_ord)

    if barajar:
        piezas_barajadas, posicion_real = barajar_piezas(piezas_ordenadas, cantidad_columnas, rng)
    else:
        # devuelve las posiciones identicas [(0,0), (0,1), ...
        piezas_barajadas = piezas_ordenadas
        posicion_real = [np.unravel_index(idx, (cantidad_filas, cantidad_columnas)) for idx in range(len(piezas_ordenadas))]

    metadatos = {
        "semilla": semilla,
        "filas": cantidad_filas,
        "columnas": cantidad_columnas,
        "barajar": barajar
    }
    if metadatos_adicionales:
        metadatos.update(metadatos_adicionales)

    return Rompecabezas(
        piezas=piezas_barajadas,
        cantidad_filas=cantidad_filas,
        cantidad_columnas=cantidad_columnas,
        posicion_real=posicion_real,
        tiene_ranuras=tiene_ranuras,
        imagen_base=img_ajustada,
        imagen_degradada=img_degradada,
        metadatos=metadatos,
    )



ESCALAS_RUIDOS = {
    "SALT_PEPPER": {
        1: (0.030, 0.030),
        2: (0.080, 0.080),
        3: (0.150, 0.150),
    },

    "GAUSSIANO": {
        1: (0.08,),
        2: (0.14,),
        3: (0.20,),
    },
 
    "UNIFORME": {
        1: (-0.14, 0.14),
        2: (-0.24, 0.24),
        3: (-0.35, 0.35),
    },

    "RAYLEIGH": {
        1: (0.0, 0.030),
        2: (0.0, 0.091),
        3: (0.0, 0.186),
    },
}

def _ruido_por_escala(tipo: str, nivel: int) -> Callable:
    """Crea una función (img, gen) -> img que aplica `tipo` de ruido al `nivel` (1, 2 o 3) de ESCALAS_RUIDOS."""
    parametros = ESCALAS_RUIDOS[tipo][nivel]
    if tipo == "SALT_PEPPER":
        probabilidad_sal, probabilidad_pimienta = parametros
        return lambda img, gen: agregar_ruido_sal_y_pimienta(
            img, gen, probabilidad_sal=probabilidad_sal, probabilidad_pimienta=probabilidad_pimienta)
    if tipo == "GAUSSIANO":
        (desviacion_estandar,) = parametros
        return lambda img, gen: agregar_ruido_gaussiano(img, gen, desviacion_estandar=desviacion_estandar)
    if tipo == "UNIFORME":
        limite_inferior, limite_superior = parametros
        return lambda img, gen: agregar_ruido_uniforme(
            img, gen, limite_inferior=limite_inferior, limite_superior=limite_superior)
    if tipo == "RAYLEIGH":
        desplazamiento, parametro_b = parametros
        return lambda img, gen: agregar_ruido_rayleigh(
            img, gen, desplazamiento=desplazamiento, parametro_b=parametro_b)
    raise ValueError(f"tipo de ruido invalido: {tipo!r}. Se espera uno de {list(ESCALAS_RUIDOS)}")


def _ruido_impulsivo(nivel_sal: Optional[int] = None, nivel_pimienta: Optional[int] = None) -> Callable:
    """
    Sal y/o pimienta, cada lado con su propio nivel de ESCALAS_RUIDOS["SALT_PEPPER"].
    Un lado en None lo deja en 0.0 (para variantes de un solo lado, como "solo sal").
    """
    probabilidad_sal = ESCALAS_RUIDOS["SALT_PEPPER"][nivel_sal][0] if nivel_sal is not None else 0.0
    probabilidad_pimienta = ESCALAS_RUIDOS["SALT_PEPPER"][nivel_pimienta][1] if nivel_pimienta is not None else 0.0
    return lambda img, gen: agregar_ruido_sal_y_pimienta(
        img, gen, probabilidad_sal=probabilidad_sal, probabilidad_pimienta=probabilidad_pimienta)


VARIANTES_NIVEL_1 = {
    "A": {
        "nombre": "Gaussiano nivel 1 + sal y pimienta nivel 1",
        "fn": componer_degradaciones(_ruido_por_escala("GAUSSIANO", 1), _ruido_impulsivo(1, 1)),
    },
    "B": {
        "nombre": "Uniforme nivel 1 + sal y pimienta nivel 1",
        "fn": componer_degradaciones(_ruido_por_escala("UNIFORME", 1), _ruido_impulsivo(1, 1)),
    },
    "C": {
        "nombre": "Gaussiano nivel 2 + SOLO SAL nivel 2",
        "fn": componer_degradaciones(_ruido_por_escala("GAUSSIANO", 2), _ruido_impulsivo(nivel_sal=2)),
    },
    "D": {
        "nombre": "Rayleigh nivel 2 + SOLO PIMIENTA nivel 2",
        "fn": componer_degradaciones(_ruido_por_escala("RAYLEIGH", 2), _ruido_impulsivo(nivel_pimienta=2)),
    },
    "E": {
        "nombre": "Uniforme nivel 3 + sal y pimienta nivel 3",
        "fn": componer_degradaciones(_ruido_por_escala("UNIFORME", 3), _ruido_impulsivo(3, 3)),
    },
    "F": {
        "nombre": "Gaussiano nivel 3 + impulsivo asimétrico (sal nivel 3, pimienta nivel 1)",
        "fn": componer_degradaciones(_ruido_por_escala("GAUSSIANO", 3), _ruido_impulsivo(nivel_sal=3, nivel_pimienta=1)),
    },
    "H": {
        "nombre": "Aleatoria: dos ruidos distintos en secuencia, cada uno con dificultad (2 o 3) sorteada por semilla",
        "fn": lambda img, gen: componer_degradaciones(*[
            _ruido_por_escala(tipo, nivel_dificultad)
            for tipo, nivel_dificultad in sorted(
                [
                    (str(tipo), int(gen.choice([2, 3])))
                    for tipo in gen.choice(list(ESCALAS_RUIDOS), size=2, replace=False)
                ],
                key=lambda par: par[0] == "SALT_PEPPER",
            )
        ])(img, gen),
    },
}

def pre_proceso_imagen_para_rompecabezas(imagen_base, filas, columnas):
    img_limpia = asegurar_rgb_float(imagen_base)
    img_ajustada = garantizar_dimensiones_para_divisibilidad(img_limpia, filas, columnas)  
    return img_ajustada

def crear_rompecabezas_nivel(
    imagen_base: np.ndarray,
    nivel: int = 1,
    filas: int = 4,
    columnas: int = 4,
    semilla: int = 42,
    **kwargs,
) -> Rompecabezas:

    rng = np.random.default_rng(semilla)
    img_ajustada = pre_proceso_imagen_para_rompecabezas(imagen_base, filas, columnas)

    if nivel == 1:
        clave_variante = str(kwargs.get("variante", "A")).upper()
        if clave_variante not in VARIANTES_NIVEL_1:
            clave_variante = "A"

        info_variante = VARIANTES_NIVEL_1[clave_variante]
        degradacion_l1 = info_variante["fn"]

        return armar_caso_rompecabezas(
            imagen_base=img_ajustada,
            cantidad_filas=filas,
            cantidad_columnas=columnas,
            degradacion_global=degradacion_l1,
            semilla=semilla,
            nivel=1,
            metadatos_adicionales={
                "variante": clave_variante,
                "nombre_ruido": info_variante["nombre"],
                "nivel": 1
            },
        )


    elif nivel == 2:
        variante_l2 = kwargs.get("variante_nivel2", None)
        if variante_l2 is None:
            variante_l2 = sortear_variante_nivel2(semilla)
        degradador_l2 = DegradacionPorPiezaNivel2(
            variante=variante_l2,
            cantidad_piezas=filas * columnas,
        )

        return armar_caso_rompecabezas(
            imagen_base=img_ajustada,
            cantidad_filas=filas,
            cantidad_columnas=columnas,
            degradacion_por_pieza=degradador_l2,
            semilla=semilla,
            metadatos_adicionales={"tipo_ruido": "fotometrico_por_pieza",
                                   "nivel": 2},
        )

    elif nivel == 3:
        degradador_l3 = TramaMixtaPorPieza()

        return armar_caso_rompecabezas(
            imagen_base=img_ajustada,
            cantidad_filas=filas,
            cantidad_columnas=columnas,
            degradacion_por_pieza=degradador_l3,
            semilla=semilla,
            nivel=3,
            metadatos_adicionales={"tipo_ruido": "trama_periodica_por_pieza",
                                   "nivel": 3},
        )

    elif nivel == 4:

        return armar_caso_rompecabezas(
                    imagen_base=img_ajustada,
                    cantidad_filas=filas,
                    cantidad_columnas=columnas,
                    semilla=semilla,
                    tiene_ranuras=True,
                    metadatos_adicionales={
                        "nivel": 4
                    }
                )

    elif nivel == 5:
        
        amplitud_rayas = kwargs.get("amplitud_rayas", 0.10)
        #TODO: CHEQUEAR ESTO
        degradador_rayas = lambda imagen, generador: imagen
        degradador_rotar = DegradacionPorPiezaRotacion()

        return armar_caso_rompecabezas(
            imagen_base=img_ajustada,
            cantidad_filas=filas,
            cantidad_columnas=columnas,
            semilla=semilla,
            tiene_ranuras=True,
            degradacion_global= degradador_rayas,
            degradacion_por_pieza=degradador_rotar,
            metadatos_adicionales={
                "nivel": 5,
                "degradador": degradador_rotar
            }
        )

  
    elif nivel == 6:
        RUIDOS_COMPATIBLES_CON_FOURIER = ("GAUSSIANO", "RAYLEIGH", "UNIFORME")
 
        #Desafios opcionales: 1 (ruido global), 2 (color), 3 (Fourier), 5 (rotacion).
        #Regla estricta: 3 y 5 NUNCA se juntan 
        candidatos = [1, 2, 3, 5]
        cantidad_problemas = int(rng.integers(1, 4))
        sorteados = rng.choice(candidatos, size=cantidad_problemas, replace=False).tolist()
 
        if 3 in sorteados and 5 in sorteados:
            sorteados.remove(int(rng.choice([3, 5])))
 
        add_global_noise = (1 in sorteados)
        add_color_degradation = (2 in sorteados)
        add_fourrier_noise = (3 in sorteados)
        add_rotation = (5 in sorteados)
 
        #1. Rayas periodicas (solo si hay rotacion, son la referencia del angulo)
        periodo_rayas = kwargs.get("periodo_rayas", 8)
        amplitud_rayas = kwargs.get("amplitud_rayas", 0.35)
        if add_rotation:
            img_base_mod = aplicar_filtro_rayas_horizontales(
                img_ajustada, periodo=periodo_rayas, amplitud=amplitud_rayas)
        else:
            img_base_mod = img_ajustada.copy()
 
        # 2. Ruido global sobre la imagen entera (Nivel 1)
        degradacion_global = None
        tipos_ruido = []
        if add_global_noise:
            pool_ruidos = (list(RUIDOS_COMPATIBLES_CON_FOURIER) if add_fourrier_noise
                           else list(ESCALAS_RUIDOS))
            tipos_ruido = rng.choice(pool_ruidos, size=2, replace=False).tolist()
 
            tipos_ruido.sort(key=lambda t: 1 if t == "SALT_PEPPER" else 0)
 
            funciones_ruido = [_ruido_por_escala(str(t), int(rng.choice([2, 3]))) for t in tipos_ruido]
            degradacion_global = componer_degradaciones(*funciones_ruido)
 
        img_degradada = img_base_mod.copy()
        if degradacion_global is not None:
            img_degradada = degradacion_global(img_degradada, rng)
 
        #3. Geometria jigsaw (Nivel 4)
        # discrete=True sortea los encastres de una tabla finita: varias costuras comparten forma exacta, 
        # asi la correlacion geometrica no alcanza sola.
        alto_img, ancho_img = img_degradada.shape[:2]
        usar_discreto = kwargs.get("discrete", True)
        jigsaw = JigsawGridGeometry(filas, columnas, alto_img, ancho_img,
                                    seed=semilla, discrete=usar_discreto)
 
        piezas_cortadas = []
        for r in range(filas):
            for c in range(columnas):
                pieza_img, _, _ = jigsaw.extract_piece_image(
                    img_degradada, r, c, padding=kwargs.get("padding", 25))
                piezas_cortadas.append(pieza_img)
 
        #4. Degradaciones por pieza: color (Nivel 2) y trama (Nivel 3)
        degradacion_por_pieza_secuencia = []
 
        variante_l2 = None
        if add_color_degradation:
            variante_l2 = kwargs.get("variante_nivel2", None)
            if variante_l2 is None:
                variante_l2 = sortear_variante_nivel2(semilla)
            degradador_l2 = DegradacionPorPiezaNivel2(
                variante=variante_l2, cantidad_piezas=filas * columnas)
            degradacion_por_pieza_secuencia.append(degradador_l2)
 
        if add_fourrier_noise:
            degradador_l3 = TramaMixtaPorPieza()
            degradacion_por_pieza_secuencia.append(degradador_l3)
 
        piezas_degradadas = []
        angulos_reales = {}
        permitir_inclinacion = kwargs.get("inclinacion_leve", True)
 
        for idx, p in enumerate(piezas_cortadas):
            adentro = (p.max(axis=2) > 0.01)
 
            p_proc = p
            for degradacion in degradacion_por_pieza_secuencia:
                p_proc = degradacion(p_proc, idx, rng)
                # Re-enmascarar DESPUES DE CADA degradacion: la trama periodica
                # suma ondas en todo el lienzo y dejaria el fondo distinto de
                # negro puro, rompiendo la deteccion de contorno del Nivel 4.
                p_proc[~adentro] = 0.0
            if not degradacion_por_pieza_secuencia:
                p_proc = p.copy()
 
            if add_rotation and permitir_inclinacion:
                jitter = float(rng.uniform(-60.0, 60.0))
                p_final, _ = enderezar_pieza(p_proc, angulo_grados=jitter, padding=0)
            else:
                jitter = 0.0
                p_final = p_proc
 
            angulos_reales[idx] = jitter
            piezas_degradadas.append(p_final)
 
        #5. Barajar
        permutacion = rng.permutation(len(piezas_degradadas))
        piezas_barajadas = [piezas_degradadas[i] for i in permutacion]
 
        posicion_real = {}
        rotacion_real = {}
        for id_nuevo, id_orig in enumerate(permutacion):
            posicion_real[id_nuevo] = (int(id_orig) // columnas, int(id_orig) % columnas)
            rotacion_real[id_nuevo] = angulos_reales[int(id_orig)]
 
        caso = Rompecabezas(
            piezas=piezas_barajadas,
            cantidad_filas=filas,
            cantidad_columnas=columnas,
            posicion_real=posicion_real,
            rotacion_real=rotacion_real if add_rotation else None,
            nivel=6,
            imagen_base=img_ajustada,
            imagen_degradada=img_degradada,
            metadatos={
                "semilla": semilla,
                "filas": filas,
                "columnas": columnas,
                "nivel": 6,
                "tipo": "jigsaw_integrador",
                "es_discreto": usar_discreto,
            },
        )
        return caso


def _buscar_directorio_imagenes_base() -> Path:
    """Busca el directorio con las imagenes base, tolerando la ruta de Colab."""
    candidatas = [
        Path("imagenes/base"),
        Path("repo_tp/imagenes/base"),
        Path(__file__).resolve().parent.parent / "imagenes" / "base",
    ]
    for c in candidatas:
        if c.exists() and any(c.glob("*.png")):
            return c
    for raiz in (Path.cwd(), Path(__file__).resolve().parent.parent):
        for p in raiz.rglob("imagenes/base"):
            if p.is_dir() and any(p.glob("*.png")):
                return p
    return Path("imagenes/base")
 
 
def generar_caso_desafio_30(
    indice_caso: int,
    semilla: int = 42,
    filas: int = 4,
    columnas: int = 4,
    directorio_imagenes: Optional[Union[str, Path]] = None,
    total_casos: int = 30,
) -> Rompecabezas:
    """
    Genera UN solo caso del desafio integrador, el numero `indice_caso`.
    """
    dir_img = Path(directorio_imagenes) if directorio_imagenes else _buscar_directorio_imagenes_base()
 
    extensiones = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"]
    archivos = []
    for ext in extensiones:
        archivos.extend(sorted(dir_img.glob(ext)))
    if not archivos:
        raise FileNotFoundError(f"No se encontraron imágenes en '{dir_img}'.")
 
    archivo_elegido = archivos[indice_caso % len(archivos)]
    img_raw = asegurar_rgb_float(
        cv2.imread(str(archivo_elegido))[:, :, ::-1].astype(np.float64) / 255.0)
 
    caso = crear_rompecabezas_nivel(
        imagen_base=img_raw,
        nivel=6,
        filas=filas,
        columnas=columnas,
        semilla=semilla + indice_caso,
    )
    caso.metadatos["archivo_origen"] = archivo_elegido.name
    caso.metadatos["id_caso"] = indice_caso + 1
    return caso
 
 
def crear_dataset_desafio_30(
    directorio_imagenes: Optional[Union[str, Path]] = None,
    filas: int = 4,
    columnas: int = 4,
    semilla_base: int = 42,
    cantidad_casos: int = 30,
) -> List[Rompecabezas]:
    """
    Construye los `cantidad_casos` rompecabezas del desafio integrador.
    """
    return [
        generar_caso_desafio_30(i, semilla=semilla_base, filas=filas,
                                columnas=columnas, directorio_imagenes=directorio_imagenes,
                                total_casos=cantidad_casos)
        for i in range(cantidad_casos)
    ]