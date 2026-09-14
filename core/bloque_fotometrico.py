# =========================================================================
# DEGRADACION FOTOMETRICA POR PIEZA (Gonzalez cap. 3 y 6) — nivel 2
# Agregar al final de core/degradaciones.py
# =========================================================================
#
# A diferencia del nivel 1 y del 3, aca no hay ruido aditivo: la imagen queda
# perfectamente nitida. Lo que cambia es que CADA PIEZA sufrio una
# transformacion puntual distinta, y por eso los valores absolutos dejan de ser
# comparables entre piezas. La ultima columna de una pieza y la primera de su
# vecina pueden venir del mismo pixel original y aun asi valer cosas muy
# distintas.
#
#
# LA IMAGEN IMPORTA MAS QUE LA DEGRADACION
# ----------------------------------------
# La solucion es ecualizar cada pieza por separado, y eso SOLO funciona si todas
# las piezas tienen estadisticas parecidas -- es decir, si la imagen es una
# textura homogenea (agua, hojas, piedras, suelo).
#
# Con contenido variado la ecualizacion falla, y falla por una razon de fondo:
# la curva de ecualizacion depende del histograma de cada pieza. Si una pieza es
# cielo y otra es piedra, se les aplican transformaciones muy distintas y dos
# bordes que deberian coincidir terminan separados.
#
# Medido con la misma degradacion sobre 6x6:
#
#     imagen              sin tocar   ecualizada
#     grass (homogenea)     21.7%       100.0%
#     gravel (homogenea)    18.3%       100.0%
#     cafe (variada)        36.7%        33.3%
#
# Por eso el nivel usa fotos de textura pareja, y por eso hay que verificar cada
# foto nueva antes de usarla.


def alterar_luminancia(pieza, gamma=1.0, contraste=1.0, brillo=0.0):
    """
    Modifica gamma, contraste y brillo dejando la CROMINANCIA intacta.

    Se trabaja en YCbCr y se toca solo el canal Y. En RGB no se puede: cualquier
    cambio de brillo aplicado a los tres canales altera tambien la saturacion.

        Y' = contraste * (Y^(1/gamma) - 0.5) + 0.5 + brillo
    """
    from skimage import color

    ycbcr = color.rgb2ycbcr(np.clip(pieza, 0.0, 1.0))

    #El canal Y de rgb2ycbcr vive en [16, 235]; se normaliza para operar
    luminancia = (ycbcr[:, :, 0] - 16.0) / 219.0
    luminancia = np.power(np.clip(luminancia, 0.0, 1.0), 1.0 / max(gamma, 0.1))
    luminancia = contraste * (luminancia - 0.5) + 0.5 + brillo

    ycbcr[:, :, 0] = np.clip(luminancia * 219.0 + 16.0, 16.0, 235.0)
    return np.clip(color.ycbcr2rgb(ycbcr), 0.0, 1.0)


def alterar_balance_rgb(pieza, ganancias, desplazamientos):
    """
    Aplica una ganancia y un offset DISTINTOS a cada canal RGB.

    Es el modelo de un balance de blancos mal ajustado: cada pieza queda con su
    propia dominante de color. A diferencia de `alterar_luminancia`, aca sí
    cambia el color, y eso se ve en el histograma como tres curvas desplazadas
    entre si.
    """
    resultado = np.empty_like(pieza)
    for indice_canal in range(3):
        resultado[:, :, indice_canal] = (pieza[:, :, indice_canal] * ganancias[indice_canal]
                                         + desplazamientos[indice_canal])
    return _acotar_rango(resultado)


VARIANTES_FOTOMETRICAS = ("luminancia", "balance_rgb")


class DegradacionFotometricaPorPieza:
    """
    Aplica a cada pieza una transformacion puntual distinta.
    Se usa como `degradacion_por_pieza`.

    Dos variantes, que se resuelven igual pero se DIAGNOSTICAN distinto:

        luminancia   gamma, contraste y brillo sobre Y. Las piezas conservan el
                     color y varian en claridad y contraste. En el histograma se
                     ve un unico bulto que se corre y se ensancha.

        balance_rgb  ganancia y offset propios en cada canal RGB. Cada pieza
                     queda con su dominante de color. En el histograma se ven
                     los tres canales separandose entre si.

    Registra en `parametros_por_pieza` que le toco a cada una, para poder
    verificar el diagnostico del alumno y no solo el resultado final.

    Los parametros se sortean sin correlacion con la posicion de la pieza: si
    piezas vecinas compartieran transformacion, comparar brillos agruparia
    piezas por su degradacion en vez de por su contenido.
    """

    def __init__(self, variante="luminancia",
                 rango_gamma=(0.55, 1.80),
                 rango_contraste=(0.60, 1.45),
                 rango_brillo=(-0.12, 0.12),
                 rango_ganancia=(0.55, 1.60),
                 rango_desplazamiento=(-0.12, 0.12)):
        if variante not in VARIANTES_FOTOMETRICAS:
            raise ValueError(f"variante invalida: {variante!r}. "
                             f"Se espera una de {VARIANTES_FOTOMETRICAS}")
        self.variante = variante
        self.rango_gamma = rango_gamma
        self.rango_contraste = rango_contraste
        self.rango_brillo = rango_brillo
        self.rango_ganancia = rango_ganancia
        self.rango_desplazamiento = rango_desplazamiento
        self.parametros_por_pieza = {}

    def __call__(self, pieza, indice, generador):
        if self.variante == "luminancia":
            gamma = float(generador.uniform(*self.rango_gamma))
            contraste = float(generador.uniform(*self.rango_contraste))
            brillo = float(generador.uniform(*self.rango_brillo))

            self.parametros_por_pieza[int(indice)] = {
                "variante": "luminancia",
                "gamma": round(gamma, 3),
                "contraste": round(contraste, 3),
                "brillo": round(brillo, 3),
            }
            return alterar_luminancia(pieza, gamma, contraste, brillo)

        ganancias = [float(generador.uniform(*self.rango_ganancia)) for _ in range(3)]
        desplazamientos = [float(generador.uniform(*self.rango_desplazamiento))
                           for _ in range(3)]

        self.parametros_por_pieza[int(indice)] = {
            "variante": "balance_rgb",
            "ganancias": [round(g, 3) for g in ganancias],
            "desplazamientos": [round(d, 3) for d in desplazamientos],
        }
        return alterar_balance_rgb(pieza, ganancias, desplazamientos)

    def reiniciar(self):
        self.parametros_por_pieza = {}

    def resumen(self):
        """Rango efectivo de los parametros sorteados, para verificar el sorteo."""
        if not self.parametros_por_pieza:
            return {}

        if self.variante == "luminancia":
            gammas = [r["gamma"] for r in self.parametros_por_pieza.values()]
            brillos = [r["brillo"] for r in self.parametros_por_pieza.values()]
            return {"variante": "luminancia",
                    "gamma": (min(gammas), max(gammas)),
                    "brillo": (min(brillos), max(brillos))}

        todas = [g for r in self.parametros_por_pieza.values() for g in r["ganancias"]]
        return {"variante": "balance_rgb", "ganancia": (min(todas), max(todas))}
