"""
Módulo de Creación y Generación de Rompecabezas Paramétricos por Nivel (1 al 5).

Representa el generador unificado oficial del curso:
- Nivel 1: Piezas cuadradas + ruido espacial (Gaussiano, Sal y Pimienta, Rayleigh, Uniforme).
- Nivel 2: Piezas cuadradas + degradación cromática por pieza (rotación de matiz o gamma/ganancia de valor).
- Nivel 3: Piezas cuadradas + trama periódica en frecuencia por pieza (ondas/productos de ondas, pensada para notch).
- Nivel 4: Piezas no cuadradas con encastres geométricos curvos (Jigsaw saliente/entrante/plano).
- Nivel 5: Piezas rotadas + filtro de rayas periódicas horizontales para estimación de orientación.
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
        DegradacionCromaticaPorPieza,
        TramaMixtaPorPieza,
    )
    from .geometria_jigsaw import JigsawGridGeometry
    from .analizador_rotacion import aplicar_filtro_rayas_horizontales, rotar_imagen_ortogonal, enderezar_pieza
except ImportError:
    from preparacion_imagenes import asegurar_rgb_float
    from degradaciones import (
        agregar_ruido_gaussiano,
        agregar_ruido_sal_y_pimienta,
        agregar_ruido_uniforme,
        agregar_ruido_rayleigh,
        componer_degradaciones,
        DegradacionCromaticaPorPieza,
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
]

class Rompecabezas:
    """
    Clase que encapsula un rompecabezas generado:
    piezas desordenadas (para los alumnos) y ground truth de auditoría (para evaluación).
    """

    def __init__(
        self,
        piezas: List[np.ndarray],
        cantidad_filas: int,
        cantidad_columnas: int,
        posicion_real: Dict[int, Tuple[int, int]],
        rotacion_real: Optional[Dict[int, float]] = None,
        nivel: int = 1,
        imagen_base: Optional[np.ndarray] = None,
        imagen_degradada: Optional[np.ndarray] = None,
        metadatos: Optional[Dict[str, Any]] = None,
    ):
        self.piezas = piezas
        self.cantidad_filas = cantidad_filas
        self.cantidad_columnas = cantidad_columnas
        self.posicion_real = posicion_real
        self.rotacion_real = rotacion_real or {p: 0.0 for p in range(len(piezas))}
        self.nivel = nivel
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
        """
        Reconstruye una imagen pegando las piezas según la grilla especificada o la solución.
        Elimina fondos negros e intercala automáticamente encastres tipo jigsaw.
        """
        if grilla_propuesta is None:
            grilla_propuesta = self.obtener_matriz_correcta()

        lista_piezas = piezas if piezas is not None else self.piezas

        # En Nivel 5, asegurar que todas las piezas queden perfectamente orientadas a 0° al reconstruir
        if self.nivel == 5:
            from core.analizador_rotacion import estimar_orientacion_fourier, enderezar_pieza
            periodo = self.metadatos.get("periodo_rayas", 8) if self.metadatos else 8
            piezas_orientadas = []
            for i, p in enumerate(lista_piezas):
                # Si tenemos la rotación real exacta para la pieza i del puzzle, usarla para alineación perfecta a 0°:
                if self.rotacion_real and i in self.rotacion_real:
                    jitter = self.rotacion_real[i]
                    p_end, _ = enderezar_pieza(self.piezas[i], angulo_grados=-jitter, padding=0)
                    piezas_orientadas.append(p_end)
                else:
                    ang = estimar_orientacion_fourier(p, periodo_esperado=periodo)
                    if abs(ang) > 0.5:
                        p_end, _ = enderezar_pieza(p, angulo_grados=ang, padding=0)
                        piezas_orientadas.append(p_end)
                    else:
                        piezas_orientadas.append(p)
            lista_piezas = piezas_orientadas

        # Dimensiones de la imagen base o deducidas de la grilla
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

        es_forma = (self.nivel in (4, 5)) or any(
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

            # Rellenar cualquier micro-costura o píxel negro residual (< 20% de la imagen)
            mascara_negra = np.all(lienzo < 0.005, axis=-1) if canales == 3 else (lienzo < 0.005)
            pct_negro = float(np.mean(mascara_negra))
            if 0 < pct_negro < 0.20:
                mask_inpaint = mascara_negra.astype(np.uint8) * 255
                lienzo_u8 = np.clip(lienzo * 255.0, 0, 255).astype(np.uint8)
                inpa_u8 = cv2.inpaint(lienzo_u8, mask_inpaint, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
                # Si aún queda algún pixel exactamente negro aislado en la máscara, segunda pasada
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
            nivel=4,
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


def armar_caso_rompecabezas(
    imagen_base: np.ndarray,
    cantidad_filas: int,
    cantidad_columnas: int,
    degradacion_global: Optional[Callable[[np.ndarray, np.random.Generator], np.ndarray]] = None,
    degradacion_por_pieza: Optional[Callable[[np.ndarray, int, np.random.Generator], np.ndarray]] = None,
    semilla: int = 42,
    nivel: int = 1,
    metadatos_adicionales: Optional[Dict[str, Any]] = None,
) -> Rompecabezas:
    """Generador base para rompecabezas con hooks de degradación."""
    rng = np.random.default_rng(semilla)
    img_limpia = asegurar_rgb_float(imagen_base)
    img_ajustada = garantizar_dimensiones_para_divisibilidad(img_limpia, cantidad_filas, cantidad_columnas)

    if degradacion_global is not None:
        img_degradada = degradacion_global(img_ajustada, rng)
    else:
        img_degradada = img_ajustada.copy()

    piezas_ordenadas = cortar_imagen_en_piezas(img_degradada, cantidad_filas, cantidad_columnas)

    if degradacion_por_pieza is not None:
        piezas_ordenadas = [
            degradacion_por_pieza(p, idx, rng) for idx, p in enumerate(piezas_ordenadas)
        ]
        grilla_ord = np.arange(cantidad_filas * cantidad_columnas).reshape(cantidad_filas, cantidad_columnas)
        img_degradada = pegar_piezas(piezas_ordenadas, grilla_ord)

    piezas_barajadas, posicion_real = barajar_piezas(piezas_ordenadas, cantidad_columnas, rng)

    metadatos = {
        "semilla": semilla,
        "filas": cantidad_filas,
        "columnas": cantidad_columnas,
        "nivel": nivel,
    }
    if metadatos_adicionales:
        metadatos.update(metadatos_adicionales)

    return Rompecabezas(
        piezas=piezas_barajadas,
        cantidad_filas=cantidad_filas,
        cantidad_columnas=cantidad_columnas,
        posicion_real=posicion_real,
        nivel=nivel,
        imagen_base=img_ajustada,
        imagen_degradada=img_degradada,
        metadatos=metadatos,
    )


# ==============================================================================
# Variantes Oficiales de Ruido para Nivel 1
# ==============================================================================

ESCALAS_RUIDOS = {
    #(probabilidad_sal, probabilidad_pimienta)
    "SALT_PEPPER" : {
        1: (0.050, 0.050),
        2: (0.115, 0.115),
        3: (0.228, 0.228),
    },

    #(desviacion_estandar,)
    "GAUSSIANO" : {
        1: (0.05,),
        2: (0.10,),
        3: (0.30,),
    },

    #(limite_inferior, limite_superior)
    "UNIFORME" : {
        1: (-0.05, 0.05),
        2: (-0.15, 0.15),
        3: (-0.35, 0.35),
    },

    #(desplazamiento, parametro_b)
    "RAYLEIGH" : {
        1: (0.0, 0.010),
        2: (0.0, 0.15),
        3: (0.0, 0.21),
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
        "fn": componer_degradaciones(_ruido_impulsivo(1, 1), _ruido_por_escala("GAUSSIANO", 1)),
    },
    "B": {
        "nombre": "Sal y pimienta nivel 1 + uniforme nivel 1",
        "fn": componer_degradaciones(_ruido_impulsivo(1, 1), _ruido_por_escala("UNIFORME", 1)),
    },
    "C": {
        "nombre": "Gaussiano nivel 2 + SOLO SAL nivel 2",
        "fn": componer_degradaciones(_ruido_impulsivo(nivel_sal=2), _ruido_por_escala("GAUSSIANO", 2)),
    },
    "D": {
        "nombre": "Rayleigh nivel 2 + SOLO PIMIENTA nivel 2",
        "fn": componer_degradaciones(_ruido_impulsivo(nivel_pimienta=2), _ruido_por_escala("RAYLEIGH", 2)),
    },
    "E": {
        "nombre": "Uniforme nivel 3 + sal y pimienta nivel 3",
        "fn": componer_degradaciones(_ruido_impulsivo(3, 3), _ruido_por_escala("UNIFORME", 3)),
    },
    "F": {
        "nombre": "Gaussiano nivel 3 + impulsivo asimétrico (sal nivel 3, pimienta nivel 1)",
        "fn": componer_degradaciones(_ruido_impulsivo(nivel_sal=3, nivel_pimienta=1), _ruido_por_escala("GAUSSIANO", 3)),
    },
    "H": {
        "nombre": "Aleatoria: dos ruidos distintos en secuencia, cada uno con dificultad (2 o 3) sorteada por semilla",
        "fn": lambda img, gen: componer_degradaciones(*(
            _ruido_por_escala(str(tipo), int(gen.choice([2, 3])))
            for tipo in gen.choice(list(ESCALAS_RUIDOS), size=2, replace=False)
        ))(img, gen),
    },
}


def crear_rompecabezas_nivel(
    imagen_base: np.ndarray,
    nivel: int = 1,
    filas: int = 4,
    columnas: int = 4,
    semilla: int = 42,
    **kwargs,
) -> Rompecabezas:
    """
    Función principal para generar rompecabezas indicando el nivel (1 a 5).

    Nivel 1:
        Piezas cuadradas con ruidos espaciales, parametrizados por ESCALAS_RUIDOS (nivel 1: leve,
        2: medio, 3: fuerte). Permite elegir 'variante' entre 'A', 'B', 'C', 'D', 'E', 'F', 'H':
        - 'A': Gaussiano nivel 1 + sal y pimienta nivel 1
        - 'B': Sal y pimienta nivel 1 + uniforme nivel 1
        - 'C': Gaussiano nivel 2 + SOLO SAL nivel 2
        - 'D': Rayleigh nivel 2 + SOLO PIMIENTA nivel 2
        - 'E': Uniforme nivel 3 + sal y pimienta nivel 3
        - 'F': Gaussiano nivel 3 + impulsivo asimétrico (sal nivel 3, pimienta nivel 1)
        - 'H': Aleatoria, dos ruidos distintos en secuencia con dificultad 2 o 3 cada uno
    Nivel 2:
        Piezas cuadradas con variaciones fotométricas por pieza (Gamma, Saturación, Brillo).
    Nivel 3:
        Piezas cuadradas con interferencia armónica periódica en frecuencia (Fourier).
    Nivel 4:
        Piezas Jigsaw con encastres curvos complementarios (Saliente, Entrante, Plano).
    Nivel 5:
        Piezas rotadas (múltiplos de 90° e inclinación leve) con filtro de modulación horizontal.
    """
    rng = np.random.default_rng(semilla)
    img_limpia = asegurar_rgb_float(imagen_base)
    img_ajustada = garantizar_dimensiones_para_divisibilidad(img_limpia, filas, columnas)

    # --------------------------------------------------------------------------
    # NIVEL 1: Ruido Espacial (6 Variantes Oficiales de la Cátedra)
    # --------------------------------------------------------------------------
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
            },
        )

    # --------------------------------------------------------------------------
    # NIVEL 2: Degradación Cromática por Pieza (rotación de matiz o gamma/ganancia de valor)
    # --------------------------------------------------------------------------
    elif nivel == 2:
        variante_cromatica = kwargs.get("variante_cromatica", "matiz")
        degradador_l2 = DegradacionCromaticaPorPieza(variante=variante_cromatica)

        return armar_caso_rompecabezas(
            imagen_base=img_ajustada,
            cantidad_filas=filas,
            cantidad_columnas=columnas,
            degradacion_por_pieza=degradador_l2,
            semilla=semilla,
            nivel=2,
            metadatos_adicionales={"tipo_ruido": "cromatico_por_pieza", "variante_cromatica": variante_cromatica},
        )

    # --------------------------------------------------------------------------
    # NIVEL 3: Trama Periódica en Frecuencia por Pieza (pensada para notch)
    # --------------------------------------------------------------------------
    elif nivel == 3:
        degradador_l3 = TramaMixtaPorPieza()

        return armar_caso_rompecabezas(
            imagen_base=img_ajustada,
            cantidad_filas=filas,
            cantidad_columnas=columnas,
            degradacion_por_pieza=degradador_l3,
            semilla=semilla,
            nivel=3,
            metadatos_adicionales={"tipo_ruido": "trama_periodica_por_pieza"},
        )

    # --------------------------------------------------------------------------
    # NIVEL 4: Piezas no cuadradas con encastres (Jigsaw Saliente/Entrante)
    # --------------------------------------------------------------------------
    elif nivel == 4:
        h, w = img_ajustada.shape[:2]
        jigsaw = JigsawGridGeometry(filas, columnas, h, w, seed=semilla)

        piezas_ordenadas = []
        for r in range(filas):
            for c in range(columnas):
                pieza_img, _, _ = jigsaw.extract_piece_image(img_ajustada, r, c, padding=kwargs.get("padding", 30))
                piezas_ordenadas.append(pieza_img)

        piezas_barajadas, posicion_real = barajar_piezas(piezas_ordenadas, columnas, rng)

        return Rompecabezas(
            piezas=piezas_barajadas,
            cantidad_filas=filas,
            cantidad_columnas=columnas,
            posicion_real=posicion_real,
            nivel=4,
            imagen_base=img_ajustada,
            imagen_degradada=img_ajustada.copy(),
            metadatos={"semilla": semilla, "filas": filas, "columnas": columnas, "nivel": 4, "tipo": "jigsaw"},
        )

    # --------------------------------------------------------------------------
    # NIVEL 5: Rotaciones + Filtro Periódico Horizontal para Detección Espectral
    # --------------------------------------------------------------------------
    elif nivel == 5:
        # 1. Aplicar modulación de rayas periódicas horizontales sobre la imagen base
        periodo_rayas = kwargs.get("periodo_rayas", 8)
        amplitud_rayas = kwargs.get("amplitud_rayas", 0.35)
        img_rayada = aplicar_filtro_rayas_horizontales(img_ajustada, periodo=periodo_rayas, amplitud=amplitud_rayas)

        # 2. Extraer piezas con geometría Jigsaw (bordes con forma analítica)
        h, w = img_ajustada.shape[:2]
        jigsaw = JigsawGridGeometry(filas, columnas, h, w, seed=semilla)

        piezas_cortadas = []
        for r in range(filas):
            for c in range(columnas):
                pieza_img, _, _ = jigsaw.extract_piece_image(img_rayada, r, c, padding=kwargs.get("padding", 30))
                piezas_cortadas.append(pieza_img)

        # 3. Rotar cada pieza con inclinación leve sobre fondo negro
        permitir_inclinacion_leve = kwargs.get("inclinacion_leve", True)

        piezas_rotadas = []
        angulos_reales = {}

        for idx, p in enumerate(piezas_cortadas):
            if permitir_inclinacion_leve:
                jitter = float(rng.uniform(-60.0, 60.0))
                p_rot, _ = enderezar_pieza(p, angulo_grados=jitter, padding=0)
            else:
                jitter = 0.0
                p_rot = p.copy()

            piezas_rotadas.append(p_rot)
            angulos_reales[idx] = jitter

        total_piezas = len(piezas_rotadas)
        permutacion = rng.permutation(total_piezas)
        piezas_barajadas = [piezas_rotadas[i] for i in permutacion]

        posicion_real = {}
        rotacion_real = {}
        for id_nuevo, id_orig in enumerate(permutacion):
            posicion_real[id_nuevo] = (int(id_orig) // columnas, int(id_orig) % columnas)
            rotacion_real[id_nuevo] = angulos_reales[int(id_orig)]

        return Rompecabezas(
            piezas=piezas_barajadas,
            cantidad_filas=filas,
            cantidad_columnas=columnas,
            posicion_real=posicion_real,
            rotacion_real=rotacion_real,
            nivel=5,
            imagen_base=img_ajustada,
            imagen_degradada=img_rayada,
            metadatos={
                "semilla": semilla,
                "filas": filas,
                "columnas": columnas,
                "nivel": 5,
                "tipo": "jigsaw_rotado",
                "periodo_rayas": periodo_rayas,
                "inclinacion_leve": permitir_inclinacion_leve,
            },
        )

    # --------------------------------------------------------------------------
    # NIVEL 6: Gran Desafío - Integración de Todos los Problemas (1, 2, 3, 4 y 5)
    # --------------------------------------------------------------------------
    elif nivel == 6:
        # 1. Modulación de rayas periódicas para deskewing (Nivel 5)
        periodo_rayas = kwargs.get("periodo_rayas", 8)
        amplitud_rayas = kwargs.get("amplitud_rayas", 0.30)
        img_mod = aplicar_filtro_rayas_horizontales(img_ajustada, periodo=periodo_rayas, amplitud=amplitud_rayas)

        # 2. Geometría Jigsaw analítica (Nivel 4)
        h, w = img_ajustada.shape[:2]
        jigsaw = JigsawGridGeometry(filas, columnas, h, w, seed=semilla)

        piezas_cortadas = []
        for r in range(filas):
            for c in range(columnas):
                pieza_img, _, _ = jigsaw.extract_piece_image(img_mod, r, c, padding=kwargs.get("padding", 25))
                piezas_cortadas.append(pieza_img)

        # 3. Seleccionar variante de ruido espacial según semilla (Nivel 1)
        variantes = ["A", "B", "C", "D", "E", "F"]
        clave_variante = str(kwargs.get("variante", variantes[semilla % len(variantes)])).upper()
        if clave_variante not in VARIANTES_NIVEL_1:
            clave_variante = "A"
        fn_ruido_l1 = VARIANTES_NIVEL_1[clave_variante]["fn"]

        variante_cromatica = kwargs.get("variante_cromatica", "matiz")
        degradador_l2 = DegradacionCromaticaPorPieza(variante=variante_cromatica)
        degradador_l3 = TramaMixtaPorPieza()

        piezas_degradadas = []
        angulos_reales = {}
        permitir_inclinacion_leve = kwargs.get("inclinacion_leve", True)

        for idx, p in enumerate(piezas_cortadas):
            mask = (p.max(axis=2) > 0.01)

            # 4.1 Degradación cromática por pieza (Nivel 2)
            p_foto = degradador_l2(p, idx, rng)
            p_foto[~mask] = 0.0

            # 4.2 Trama periódica en frecuencia por pieza (Nivel 3)
            p_trama = degradador_l3(p_foto, idx, rng)
            p_trama[~mask] = 0.0

            # 4.3 Ruido espacial mixto (Nivel 1)
            p_ruidosa = fn_ruido_l1(p_trama, rng)
            p_ruidosa[~mask] = 0.0

            # 4.4 Rotación con inclinación aleatoria (Nivel 5)
            if permitir_inclinacion_leve:
                jitter = float(rng.uniform(-10.0, 10.0))
                p_rot, _ = enderezar_pieza(p_ruidosa, angulo_grados=jitter, padding=0)
            else:
                jitter = 0.0
                p_rot = p_ruidosa.copy()

            piezas_degradadas.append(p_rot)
            angulos_reales[idx] = jitter

        # 5. Barajar piezas
        total_piezas = len(piezas_degradadas)
        permutacion = rng.permutation(total_piezas)
        piezas_barajadas = [piezas_degradadas[i] for i in permutacion]

        posicion_real = {}
        rotacion_real = {}
        for id_nuevo, id_orig in enumerate(permutacion):
            posicion_real[id_nuevo] = (int(id_orig) // columnas, int(id_orig) % columnas)
            rotacion_real[id_nuevo] = angulos_reales[int(id_orig)]

        return Rompecabezas(
            piezas=piezas_barajadas,
            cantidad_filas=filas,
            cantidad_columnas=columnas,
            posicion_real=posicion_real,
            rotacion_real=rotacion_real,
            nivel=6,
            imagen_base=img_ajustada,
            imagen_degradada=img_mod,
            metadatos={
                "semilla": semilla,
                "filas": filas,
                "columnas": columnas,
                "nivel": 6,
                "tipo": "multidegradado_integrador",
                "variante_ruido_espacial": clave_variante,
                "nombre_ruido_espacial": VARIANTES_NIVEL_1[clave_variante]["nombre"],
                "variante_cromatica": variante_cromatica,
                "periodo_rayas": periodo_rayas,
                "inclinacion_leve": permitir_inclinacion_leve,
            },
        )

    else:
        raise ValueError(f"Nivel no válido: {nivel}. Debe ser un entero entre 1 y 6.")


def crear_dataset_desafio_30(
    directorio_imagenes: Union[str, Path],
    directorio_salida: Optional[Union[str, Path]] = None,
    filas: int = 10,
    columnas: int = 10,
    semilla_base: int = 1000,
    cantidad_casos: int = 30,
) -> List[Rompecabezas]:
    """
    Crea y configura el dataset de 30 rompecabezas integradores de 10x10 a partir de
    las imágenes de la carpeta especificada, asignando semillas y problemas únicos a cada caso.
    """
    dir_img = Path(directorio_imagenes)
    extensiones = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"]
    archivos = []
    for ext in extensiones:
        archivos.extend(sorted(dir_img.glob(ext)))

    if not archivos:
        raise FileNotFoundError(f"No se encontraron imágenes en el directorio '{dir_img}'.")

    casos_dataset = []
    print(f"[Dataset] Generando {cantidad_casos} rompecabezas de {filas}x{columnas} ({filas * columnas} piezas c/u)...")

    for i in range(cantidad_casos):
        archivo_elegido = archivos[i % len(archivos)]
        img_raw = asegurar_rgb_float(cv2.imread(str(archivo_elegido))[:, :, ::-1].astype(np.float64) / 255.0)

        semilla_caso = semilla_base + i
        caso = crear_rompecabezas_nivel(
            imagen_base=img_raw,
            nivel=6,
            filas=filas,
            columnas=columnas,
            semilla=semilla_caso,
        )
        caso.metadatos["archivo_origen"] = archivo_elegido.name
        caso.metadatos["id_caso"] = i + 1
        casos_dataset.append(caso)

        if (i + 1) % 5 == 0 or (i + 1) == cantidad_casos:
            print(f"  -> Caso {i + 1:2d}/{cantidad_casos}: [{archivo_elegido.name}] - Var Ruido: {caso.metadatos['variante_ruido_espacial']} (Semilla {semilla_caso})")

    return casos_dataset
