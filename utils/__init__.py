"""
Módulo de Utilidades para Visualización y Animación.
"""

from .visualizacion import (
    mostrar_piezas_desordenadas,
    mostrar_comparacion_imagen,
    mostrar_espectro_fourier,
    mostrar_reconstruccion,
    mostrar_matriz_afinidad,
)
from .animacion import (
    crear_animacion,
    renderizar_pasos,
)
from .visualizacion_v2 import (
    mostrar_piezas_desordenadas_v2,
    mostrar_reconstruccion_v2,
    generar_reporte_completo_v2,
    renderizar_pasos_v2,
    crear_animacion_v2,
)

__all__ = [
    "mostrar_piezas_desordenadas",
    "mostrar_comparacion_imagen",
    "mostrar_espectro_fourier",
    "mostrar_reconstruccion",
    "mostrar_matriz_afinidad",
    "crear_animacion",
    "renderizar_pasos",
    "mostrar_piezas_desordenadas_v2",
    "mostrar_reconstruccion_v2",
    "generar_reporte_completo_v2",
    "renderizar_pasos_v2",
    "crear_animacion_v2",
]
