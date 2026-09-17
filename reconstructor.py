"""
Módulo de Reconstrucción del Rompecabezas a partir de Afinidades / Métricas de Compatibilidad.

Características principales:
1. Tablero expandible de (2F - 1) x (2C - 1) para permitir crecimiento libre del fragmento.
2. Búsqueda Best-First con retroceso acotado (Backtracking) y compleción voraz (Greedy).
3. Universalidad: Funciona con CUALQUIER función de compatibilidad o matrices de afinidad
   suministradas por los estudiantes.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union, Any
import numpy as np

try:
    from .bordes import construir_matrices_afinidad, compatibilidad_baseline
except ImportError:
    from bordes import construir_matrices_afinidad, compatibilidad_baseline

__all__ = [
    "CELDA_VACIA",
    "ReconstructorRompecabezas",
    "reconstruir_desde_afinidades",
    "reconstruir_rompecabezas",
]

CELDA_VACIA = -1


class ReconstructorRompecabezas:
    """
    Reconstructor del rompecabezas a partir de matrices de afinidad (costo de ensamble).
    """

    def __init__(
        self,
        matrices_afinidad: Dict[str, np.ndarray],
        cantidad_filas: int,
        cantidad_columnas: int,
        ancho_haz: int = 3,
        maximo_retrocesos: int = 500,
    ):
        self.afinidad_horizontal = np.asarray(matrices_afinidad["horizontal"], dtype=np.float64)
        self.afinidad_vertical = np.asarray(matrices_afinidad["vertical"], dtype=np.float64)

        self.cantidad_piezas = self.afinidad_horizontal.shape[0]
        self.cantidad_filas = cantidad_filas
        self.cantidad_columnas = cantidad_columnas
        self.ancho_haz = ancho_haz
        self.maximo_retrocesos = maximo_retrocesos

        piezas_esperadas = cantidad_filas * cantidad_columnas
        if self.cantidad_piezas != piezas_esperadas:
            raise ValueError(
                f"Las matrices tienen {self.cantidad_piezas} piezas pero la grilla "
                f"{cantidad_filas}x{cantidad_columnas} requiere {piezas_esperadas}."
            )

        self.alto_tablero = 2 * cantidad_filas - 1
        self.ancho_tablero = 2 * cantidad_columnas - 1
        self.fila_centro = cantidad_filas - 1
        self.columna_centro = cantidad_columnas - 1

        self.historial_colocaciones = []
        self.grilla_resultado = None
        self.costo_total = float("inf")

        self.tablero = np.array([])
        self.piezas_usadas = set()
        self.limites = None
        self.retrocesos = 0
        self.camino_actual = []

        self.mejor_tablero = None
        self.mejor_limites = None
        self.mejor_costo = float("inf")
        self.mejor_camino = []

    def _reiniciar_estado(self) -> None:
        self.tablero = np.full((self.alto_tablero, self.ancho_tablero), CELDA_VACIA, dtype=int)
        self.piezas_usadas = set()
        self.limites = None
        self.retrocesos = 0
        self.camino_actual = []

        self.mejor_tablero = None
        self.mejor_limites = None
        self.mejor_costo = float("inf")
        self.mejor_camino = []

    def _colocar_pieza(self, fila: int, columna: int, id_pieza: int, costo: float) -> Optional[Tuple[int, int, int, int]]:
        limites_previos = self.limites

        self.tablero[fila, columna] = id_pieza
        self.piezas_usadas.add(id_pieza)

        if limites_previos is None:
            self.limites = (fila, fila, columna, columna)
        else:
            f_min, f_max, c_min, c_max = limites_previos
            self.limites = (min(f_min, fila), max(f_max, fila), min(c_min, columna), max(c_max, columna))

        self.camino_actual.append({
            "paso": len(self.piezas_usadas),
            "id_pieza": int(id_pieza),
            "fila_tablero": int(fila),
            "columna_tablero": int(columna),
            "costo": float(costo),
        })
        return limites_previos

    def _deshacer_movimiento(self, fila: int, columna: int, id_pieza: int, limites_previos) -> None:
        self.tablero[fila, columna] = CELDA_VACIA
        self.piezas_usadas.discard(id_pieza)
        self.limites = limites_previos
        if self.camino_actual:
            self.camino_actual.pop()

    def _cabe_en_la_grilla(self, fila: int, columna: int) -> bool:
        if self.limites is None:
            return True

        f_min, f_max, c_min, c_max = self.limites
        alto = max(f_max, fila) - min(f_min, fila) + 1
        ancho = max(c_max, columna) - min(c_min, columna) + 1

        return alto <= self.cantidad_filas and ancho <= self.cantidad_columnas

    def _celdas_frontera(self) -> List[Tuple[int, int]]:
        """
        Identifica todas las posiciones vacías adyacentes al bloque de piezas ya colocadas.

        Filtra el tablero para retornar únicamente las celdas disponibles que no excedan
        las dimensiones máximas permitidas para el rompecabezas final (filas x columnas).

        Returns:
            List[Tuple[int, int]]: Lista de coordenadas (fila, columna) candidatas.
        """
        #Si no hay piezas colocadas en el tablero, no existe frontera
        if self.limites is None:
            return []

        f_min, f_max, c_min, c_max = self.limites
        frontera = []

        rango_filas = range(max(0, f_min - 1), min(self.alto_tablero, f_max + 2))
        rango_columnas = range(max(0, c_min - 1), min(self.ancho_tablero, c_max + 2))

        for fila in rango_filas:
            for columna in rango_columnas:
                if self.tablero[fila, columna] != CELDA_VACIA:
                    continue

                if not self._cabe_en_la_grilla(fila, columna):
                    continue

                tiene_vecino_izq = columna > 0 and self.tablero[fila, columna - 1] != CELDA_VACIA
                tiene_vecino_der = columna + 1 < self.ancho_tablero and self.tablero[fila, columna + 1] != CELDA_VACIA
                tiene_vecino_arriba = fila > 0 and self.tablero[fila - 1, columna] != CELDA_VACIA
                tiene_vecino_abajo = fila + 1 < self.alto_tablero and self.tablero[fila + 1, columna] != CELDA_VACIA

                if tiene_vecino_izq or tiene_vecino_der or tiene_vecino_arriba or tiene_vecino_abajo:
                    frontera.append((fila, columna))

        return frontera

    def _mejores_colocaciones(self, cantidad: int) -> List[Tuple[float, int, int, int]]:
        """
        Selección Best-First: Evalúa todas las combinaciones posibles de (celda_frontera, pieza_libre)
        y retorna las N mejores según su compatibilidad.

        Genera el producto cartesiano entre las casillas disponibles en el borde y las piezas que
        aún no han sido colocadas, calcula el costo de calce para cada par y filtra los mejores candidatos.

        Vectorizado con numpy: para cada celda de la frontera se calcula el costo contra
        TODAS las piezas libres de una sola vez (indexado de arrays en vez de un loop
        Python por par celda-pieza), y la seleccion de los `cantidad` mejores usa
        argpartition en vez de ordenar la lista completa de candidatos. Con grillas
        grandes (p. ej. 16x16 = 256 piezas) el loop Python par-por-par se vuelve el
        cuello de botella dominante de toda la reconstruccion.

        Args:
            cantidad (int): Número máximo de candidatos a retornar (delimitado por el ancho_haz).

        Returns:
            List[Tuple[float, int, int, int]]: Lista de tuplas con la estructura:
                (costo_compatibilidad, fila, columna, id_pieza), ordenada de menor costo a mayor.
        """
        celdas = self._celdas_frontera()
        piezas_libres = np.array(
            [p for p in range(self.cantidad_piezas) if p not in self.piezas_usadas], dtype=int)

        if not celdas or piezas_libres.size == 0:
            return []

        #Costo de cada pieza libre en cada celda de la frontera, todas las piezas
        #a la vez por celda (misma formula que antes, solo que vectorizada)
        costos_por_celda = []
        for fila, columna in celdas:
            costo = np.zeros(piezas_libres.shape[0], dtype=np.float64)

            if columna > 0:
                vecino_izq = self.tablero[fila, columna - 1]
                if vecino_izq != CELDA_VACIA:
                    costo += self.afinidad_horizontal[vecino_izq, piezas_libres]

            if columna + 1 < self.ancho_tablero:
                vecino_der = self.tablero[fila, columna + 1]
                if vecino_der != CELDA_VACIA:
                    costo += self.afinidad_horizontal[piezas_libres, vecino_der]

            if fila > 0:
                vecino_arriba = self.tablero[fila - 1, columna]
                if vecino_arriba != CELDA_VACIA:
                    costo += self.afinidad_vertical[vecino_arriba, piezas_libres]

            if fila + 1 < self.alto_tablero:
                vecino_abajo = self.tablero[fila + 1, columna]
                if vecino_abajo != CELDA_VACIA:
                    costo += self.afinidad_vertical[piezas_libres, vecino_abajo]

            costos_por_celda.append(costo)

        #Aplanamos: mismo orden (celda, luego pieza) que generaba el doble loop original
        costos = np.concatenate(costos_por_celda)
        filas_candidatas = np.repeat([fila for fila, _ in celdas], piezas_libres.shape[0])
        columnas_candidatas = np.repeat([columna for _, columna in celdas], piezas_libres.shape[0])
        piezas_candidatas = np.tile(piezas_libres, len(celdas))

        #Los `cantidad` mejores sin ordenar TODOS los candidatos: argpartition separa
        #los mejores en O(n), y solo esos pocos se terminan de ordenar
        cantidad = min(cantidad, costos.shape[0])
        indices_mejores = np.argpartition(costos, cantidad - 1)[:cantidad]
        indices_mejores = indices_mejores[np.argsort(costos[indices_mejores], kind="stable")]

        return [
            (float(costos[indice]), int(filas_candidatas[indice]),
             int(columnas_candidatas[indice]), int(piezas_candidatas[indice]))
            for indice in indices_mejores
        ]

    def _buscar_backtracking(self, costo_acumulado: float) -> bool:
        if len(self.piezas_usadas) == self.cantidad_piezas:
            self.mejor_costo = costo_acumulado
            self.mejor_tablero = self.tablero.copy()
            self.mejor_limites = self.limites
            self.mejor_camino = [dict(paso) for paso in self.camino_actual]
            return True

        if self.retrocesos >= self.maximo_retrocesos:
            return False

        candidatos = self._mejores_colocaciones(self.ancho_haz)
        if not candidatos:
            return False

        for costo, fila, columna, id_pieza in candidatos:
            limites_previos = self._colocar_pieza(fila, columna, id_pieza, costo)

            if self._buscar_backtracking(costo_acumulado + costo):
                return True

            self._deshacer_movimiento(fila, columna, id_pieza, limites_previos)
            self.retrocesos += 1

            if self.retrocesos >= self.maximo_retrocesos:
                break

        return False

    def _completar_con_greedy(self) -> None:
        while len(self.piezas_usadas) < self.cantidad_piezas:
            candidatos = self._mejores_colocaciones(cantidad=1)
            if not candidatos:
                # Si se estancó, colocar en la primera celda libre con menor costo aunque sea infinito
                piezas_libres = [p for p in range(self.cantidad_piezas) if p not in self.piezas_usadas]
                fronteras = self._celdas_frontera()
                if not fronteras or not piezas_libres:
                    break
                f, c = fronteras[0]
                p = piezas_libres[0]
                self._colocar_pieza(f, c, p, 1e6)
                continue

            costo, fila, columna, id_pieza = candidatos[0]
            self._colocar_pieza(fila, columna, id_pieza, costo)

    def _calcular_origen_recorte(self, limites) -> Tuple[int, int]:
        fila_minima, _, columna_minima, _ = limites
        max_fila_origen = self.alto_tablero - self.cantidad_filas
        fila_inicio = max(0, min(fila_minima, max_fila_origen))

        max_columna_origen = self.ancho_tablero - self.cantidad_columnas
        columna_inicio = max(0, min(columna_minima, max_columna_origen))

        return fila_inicio, columna_inicio

    def _recortar_al_bounding_box(self, tablero: np.ndarray, limites) -> np.ndarray:
        fila_inicio, columna_inicio = self._calcular_origen_recorte(limites)
        fila_fin = fila_inicio + self.cantidad_filas
        columna_fin = columna_inicio + self.cantidad_columnas
        return tablero[fila_inicio:fila_fin, columna_inicio:columna_fin].copy()

    def _traducir_historial(self, camino: List[Dict[str, Any]], limites) -> List[Dict[str, Any]]:
        fila_origen, columna_origen = self._calcular_origen_recorte(limites)
        historial_traducido = []

        for paso in camino:
            fila_grilla = paso["fila_tablero"] - fila_origen
            columna_grilla = paso["columna_tablero"] - columna_origen

            esta_dentro = 0 <= fila_grilla < self.cantidad_filas and 0 <= columna_grilla < self.cantidad_columnas
            if not esta_dentro:
                continue

            nuevo_paso = dict(paso)
            nuevo_paso["fila"] = int(fila_grilla)
            nuevo_paso["columna"] = int(columna_grilla)
            historial_traducido.append(nuevo_paso)

        for indice, paso in enumerate(historial_traducido, start=1):
            paso["paso"] = indice

        return historial_traducido

    def _obtener_pares_iniciales(self, cantidad_pares: int) -> List[Tuple[int, int]]:
        matriz = self.afinidad_horizontal
        indices_planos = np.argsort(matriz, axis=None)[:cantidad_pares]
        pares_iniciales = []
        for indice in indices_planos:
            pieza_a, pieza_b = np.unravel_index(indice, matriz.shape)
            pares_iniciales.append((int(pieza_a), int(pieza_b)))
        return pares_iniciales

    def reconstruir(
        self,
        cantidad_semillas: int = 3,
        pieza_ancla: Optional[Tuple[int, int, int]] = None,
    ) -> np.ndarray:
        """
        Ejecuta la búsqueda de reconstrucción probando las semillas de menor costo.
        """
        mejor_grilla_global = None
        mejor_costo_global = float("inf")
        mejor_camino_global = []

        if pieza_ancla is not None:
            semillas = [(pieza_ancla[0], None)]
        else:
            semillas = self._obtener_pares_iniciales(cantidad_semillas)
            if not semillas:
                semillas = [(0, 1)]

        for semilla in semillas:
            self._reiniciar_estado()

            if pieza_ancla is not None:
                id_pieza, fila_final, columna_final = pieza_ancla
                self._colocar_pieza(self.fila_centro + fila_final, self.columna_centro + columna_final, id_pieza, costo=0.0)
                costo_inicial = 0.0
            else:
                pieza_a, pieza_b = semilla
                self._colocar_pieza(self.fila_centro, self.columna_centro, pieza_a, costo=0.0)
                costo_inicial = float(self.afinidad_horizontal[pieza_a, pieza_b])
                self._colocar_pieza(self.fila_centro, self.columna_centro + 1, pieza_b, costo_inicial)

            self.mejor_tablero = self.tablero.copy()
            self.mejor_limites = self.limites
            self.mejor_camino = [dict(paso) for paso in self.camino_actual]

            self._buscar_backtracking(costo_inicial)

            if self.mejor_tablero is not None:
                self.tablero = self.mejor_tablero.copy()
                self.limites = self.mejor_limites
                self.piezas_usadas = {int(p) for p in self.tablero[self.tablero != CELDA_VACIA]}
                self.camino_actual = [dict(paso) for paso in self.mejor_camino]

            self._completar_con_greedy()

            costo_final = self.mejor_costo if np.isfinite(self.mejor_costo) else float("inf")
            if costo_final < mejor_costo_global or mejor_grilla_global is None:
                if self.limites is not None:
                    mejor_costo_global = costo_final
                    mejor_grilla_global = self._recortar_al_bounding_box(self.tablero, self.limites)
                    mejor_camino_global = self._traducir_historial(self.camino_actual, self.limites)

        self.grilla_resultado = mejor_grilla_global
        self.costo_total = mejor_costo_global
        self.historial_colocaciones = mejor_camino_global

        return mejor_grilla_global


def reconstruir_desde_afinidades(
    matrices_afinidad: Dict[str, np.ndarray],
    cantidad_filas: int,
    cantidad_columnas: int,
    ancho_haz: int = 3,
    maximo_retrocesos: int = 500,
    cantidad_semillas: int = 3,
    pieza_ancla: Optional[Tuple[int, int, int]] = None,
    devolver_reconstructor: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, ReconstructorRompecabezas]]:
    """
    Función de ensamble estándar dada la matriz de afinidades.
    """
    reconstructor = ReconstructorRompecabezas(
        matrices_afinidad=matrices_afinidad,
        cantidad_filas=cantidad_filas,
        cantidad_columnas=cantidad_columnas,
        ancho_haz=ancho_haz,
        maximo_retrocesos=maximo_retrocesos,
    )
    grilla = reconstructor.reconstruir(cantidad_semillas=cantidad_semillas, pieza_ancla=pieza_ancla)

    if devolver_reconstructor:
        return grilla, reconstructor
    return grilla


def reconstruir_rompecabezas(
    piezas: List[np.ndarray],
    cantidad_filas: int,
    cantidad_columnas: int,
    funcion_compatibilidad: Optional[Callable[[np.ndarray, np.ndarray, str], float]] = None,
    matrices_afinidad: Optional[Dict[str, np.ndarray]] = None,
    ancho_haz: int = 3,
    maximo_retrocesos: int = 500,
    cantidad_semillas: int = 3,
    pieza_ancla: Optional[Tuple[int, int, int]] = None,
    devolver_reconstructor: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, ReconstructorRompecabezas]]:
    """
    Función de reconstrucción universal de alto nivel.
    Acepta CUALQUIER función de compatibilidad provista por los alumnos:
        funcion_compatibilidad(pieza_a, pieza_b, relacion="horizontal"|"vertical") -> float (costo)

    Si no se especifica ninguna métrica ni matriz, utiliza `compatibilidad_baseline`.
    """
    if matrices_afinidad is None:
        if funcion_compatibilidad is None:
            funcion_compatibilidad = compatibilidad_baseline
        matrices_afinidad = construir_matrices_afinidad(piezas, funcion_compatibilidad=funcion_compatibilidad)

    return reconstruir_desde_afinidades(
        matrices_afinidad=matrices_afinidad,
        cantidad_filas=cantidad_filas,
        cantidad_columnas=cantidad_columnas,
        ancho_haz=ancho_haz,
        maximo_retrocesos=maximo_retrocesos,
        cantidad_semillas=cantidad_semillas,
        pieza_ancla=pieza_ancla,
        devolver_reconstructor=devolver_reconstructor,
    )
