"""
Módulo Central (Core) del TP Rompecabezas.
Infraestructura completa para generar, evaluar y resolver rompecabezas en 5 niveles.
"""

from .preparacion_imagenes import (
    cargar_imagen,
    guardar_imagen,
    preparar_imagen_base,
    asegurar_rgb_float,
    float_a_uint8,
)
from .degradaciones import (
    agregar_ruido_gaussiano,
    agregar_ruido_uniforme,
    agregar_ruido_rayleigh,
    agregar_ruido_sal_y_pimienta,
    rotar_matiz,
    alterar_valor,
    VARIANTES_CROMATICAS,
    DegradacionCromaticaPorPieza,
    agregar_onda,
    agregar_ondas,
    agregar_producto_de_ondas,
    TIPOS_DE_TRAMA,
    TramaMixtaPorPieza,
    componer_degradaciones,
)
from .geometria_jigsaw import (
    generate_tab_curve,
    JigsawGridGeometry,
    TABLA_BORDES_DISCRETOS,
)
from .detector_forma import (
    binarize_piece,
    extract_external_contour,
    detect_jigsaw_corners,
    detect_corners_and_split_sides,
    segmentar_lados_pieza,
    segmentar_borde_en_4,
    pasar_borde_a_1d,
    extraer_perfil_1d,
    analyze_piece_shape,
    compute_edge_correlation,
    compute_jigsaw_shape_compatibility,
    calcular_mse_color_bordes,
)
from .analizador_rotacion import (
    aplicar_filtro_rayas_horizontales,
    estimar_orientacion_fourier,
    estimar_orientacion_sobel,
    enderezar_pieza,
    rotar_imagen_ortogonal,
)
from .bordes import (
    LADOS,
    BORDES_ENFRENTADOS,
    extraer_banda_borde,
    compatibilidad_baseline,
    construir_matrices_afinidad,
)
from .metricas import (
    calcular_psnr,
    calcular_ssim,
    calcular_precision_top1,
    calcular_rango_reciproco_medio,
    calcular_precision_directa,
    calcular_precision_vecindad,
    generar_reporte_completo,
    imprimir_reporte,
)
from .reconstructor import (
    CELDA_VACIA,
    ReconstructorRompecabezas,
    reconstruir_desde_afinidades,
    reconstruir_rompecabezas,
)
from .crear_rompecabezas import (
    Rompecabezas,
    cortar_imagen_en_piezas,
    barajar_piezas,
    pegar_piezas,
    armar_caso_rompecabezas,
    crear_rompecabezas_nivel,
    crear_dataset_desafio_30,
)

__all__ = [
    # Preparación
    "cargar_imagen", "guardar_imagen", "preparar_imagen_base", "asegurar_rgb_float", "float_a_uint8",
    # Degradaciones
    "agregar_ruido_gaussiano", "agregar_ruido_uniforme", "agregar_ruido_rayleigh", "agregar_ruido_sal_y_pimienta",
    "rotar_matiz", "alterar_valor", "VARIANTES_CROMATICAS", "DegradacionCromaticaPorPieza",
    "agregar_onda", "agregar_ondas", "agregar_producto_de_ondas", "TIPOS_DE_TRAMA", "TramaMixtaPorPieza",
    "componer_degradaciones",
    # Jigsaw
    "generate_tab_curve", "JigsawGridGeometry", "TABLA_BORDES_DISCRETOS",
    # Forma
    "binarize_piece", "extract_external_contour", "detect_jigsaw_corners", "detect_corners_and_split_sides",
    "segmentar_lados_pieza", "segmentar_borde_en_4", "pasar_borde_a_1d", "extraer_perfil_1d", "analyze_piece_shape",
    "compute_edge_correlation", "compute_jigsaw_shape_compatibility", "calcular_mse_color_bordes",
    # Rotación
    "aplicar_filtro_rayas_horizontales", "estimar_orientacion_fourier", "estimar_orientacion_sobel", "enderezar_pieza", "rotar_imagen_ortogonal",
    # Bordes y afinidades
    "LADOS", "BORDES_ENFRENTADOS", "extraer_banda_borde", "compatibilidad_baseline", "construir_matrices_afinidad",
    # Métricas
    "calcular_psnr", "calcular_ssim", "calcular_precision_top1", "calcular_rango_reciproco_medio",
    "calcular_precision_directa", "calcular_precision_vecindad", "generar_reporte_completo", "imprimir_reporte",
    # Reconstructor
    "CELDA_VACIA", "ReconstructorRompecabezas", "reconstruir_desde_afinidades", "reconstruir_rompecabezas",
    # Generador
    "Rompecabezas", "cortar_imagen_en_piezas", "barajar_piezas", "pegar_piezas", "armar_caso_rompecabezas", "crear_rompecabezas_nivel", "crear_dataset_desafio_30",
]
