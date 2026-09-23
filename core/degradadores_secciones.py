from core.degradacionesV2 import (
    DegradadorGaussiano,
    DegradadorSalYPimienta,
    DegradadorRayleigh,
    DegradadorUniforme,
    DegradadorAleatorioGlobal,
    DegradadorCompuesto,
    DegradadorPorCrominancia,
    DegradadorPorLuminancia,
    DegradadorBalanceado,
    DegradadorFourierCuadruple,
    DegradadorFourierDiagonal,
    DegradadorFourierDobleFrecuencia,
    DegradadorFourierOblicuo,
    DegradadorFourierOrtogonal,
    DegradadorAleatorio,
    DegradadorRotacionYLineasAlMismoAngulo,
    DegradadorPorPiezaCompuesto,
)

#region seccion 1

degradadorSP_1 = DegradadorSalYPimienta(0.030, 0.030)
degradadorSP_2 = DegradadorSalYPimienta(0.080, 0.080)
degradadorSP_3 = DegradadorSalYPimienta(0.150, 0.150)
degradadorSP_2_3 = DegradadorAleatorioGlobal(degradadorSP_2,degradadorSP_3)

degradadorGaussiano_1 = DegradadorGaussiano(desviacion_estandar=0.08)
degradadorGaussiano_2 = DegradadorGaussiano(desviacion_estandar=0.14)
degradadorGaussiano_3 = DegradadorGaussiano(desviacion_estandar=0.20)
degradadorGaussiano_2_3 = DegradadorAleatorioGlobal(degradadorGaussiano_2,degradadorGaussiano_3)

degradadorUniforme_1 = DegradadorUniforme(-0.14, 0.14)
degradadorUniforme_2 = DegradadorUniforme(-0.24, 0.24)
degradadorUniforme_3 = DegradadorUniforme(-0.35, 0.35)
degradadorUniforme_2_3 = DegradadorAleatorioGlobal(degradadorUniforme_2, degradadorUniforme_3)

degradadorRayleigh_1 = DegradadorRayleigh(0.0, 0.030)
degradadorRayleigh_2 = DegradadorRayleigh(0.0, 0.091)
degradadorRayleigh_3 = DegradadorRayleigh(0.0, 0.186)
degradadorRayleigh_2_3 = DegradadorAleatorioGlobal(degradadorRayleigh_2, degradadorRayleigh_3)

degradador_ruido_uniforme = DegradadorAleatorioGlobal(degradadorGaussiano_2_3, degradadorUniforme_2_3, degradadorRayleigh_2_3)

degradador_compuest_uniforme_SP = DegradadorCompuesto(degradador_ruido_uniforme, degradadorSP_2_3)
degradador_ruido_uniforme_uniforme = DegradadorCompuesto(degradador_ruido_uniforme, degradador_ruido_uniforme)

degradador_seccion_1 = DegradadorAleatorioGlobal(degradador_compuest_uniforme_SP, degradador_ruido_uniforme_uniforme)

#end region

#region seccion 2

degradadorLuminancia = DegradadorPorLuminancia(gamma=(0.55, 1.80), contraste=(0.60, 1.45), brillo=(-0.12, 0.12))
degradadorPorCrominancia = DegradadorPorCrominancia(ganancia_cb=(0.30, 2.00), ganancia_cr=(0.30, 2.00), desplazamiento_cb=(-45.0, 45.0), desplazamiento_cr=(-45.0, 45.0))
degradadorPorXoR = DegradadorBalanceado(degradadorLuminancia, degradadorPorCrominancia, cantidad_de_piezas=6*6)

#end region

#region seccion 3

FRECUENCIAS_DISPONIBLES = (24,  40,  56, 100)
AMPLITUD_RANGO = (0.25,0.35)

degradadorCuadruple = DegradadorFourierCuadruple(FRECUENCIAS_DISPONIBLES, amplitud=AMPLITUD_RANGO)
degradadorDiagonal = DegradadorFourierDiagonal(FRECUENCIAS_DISPONIBLES, amplitud=AMPLITUD_RANGO)
degradadorDobleFrecuencia = DegradadorFourierDobleFrecuencia(FRECUENCIAS_DISPONIBLES, amplitud=AMPLITUD_RANGO)
degradadorOblicuo = DegradadorFourierOblicuo(FRECUENCIAS_DISPONIBLES, amplitud=AMPLITUD_RANGO)
degradadorOrtogonal = DegradadorFourierOrtogonal(FRECUENCIAS_DISPONIBLES, amplitud=AMPLITUD_RANGO)

degradador_seccion_3 = DegradadorAleatorio(degradadorCuadruple, degradadorDiagonal, degradadorDobleFrecuencia, degradadorOblicuo, degradadorOrtogonal)

#end region

#region seccion 5

degradador_seccion_5 = DegradadorRotacionYLineasAlMismoAngulo(angulo=(5.0, 60.0), amplitud=0.3)

#end region

#region combinaciones

# Color + Fourier, en ese orden: primero se distorsiona luminancia/crominancia y despues se
# suma la trama periodica sobre el color ya distorsionado.
degradador_color_fourier = DegradadorPorPiezaCompuesto(degradadorPorXoR, degradador_seccion_3)

# Color + Rotacion (con lineas al mismo angulo), en ese orden: se tine la pieza y despues se
# rota junto con las lineas de referencia.
degradador_color_rotacion = DegradadorPorPiezaCompuesto(degradadorPorXoR, degradador_seccion_5)

# Mismo ruido continuo que compone degradador_seccion_1, pero sin sal y pimienta: se usa en las
# combinaciones de 3 o mas degradaciones (seccion 1 + color + fourier [+ geometria/rotacion]),
# donde sumarle ademas ruido impulsivo ya no deja superar el 85% de precision de vecindad.
degradador_global_sin_impulsivo = degradador_ruido_uniforme

#end region

