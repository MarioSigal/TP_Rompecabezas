# =========================================================================
# DEGRADACION FOTOMETRICA POR PIEZA  — nivel 2
# =========================================================================

def alterar_luminancia(pieza, gamma=1.0, contraste=1.0, brillo=0.0):
    """
    Modifica gamma, contraste y brillo.
 
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
    propia dominante de color.
    """
    resultado = np.empty_like(pieza)
    for indice_canal in range(3):
        resultado[:, :, indice_canal] = (pieza[:, :, indice_canal] * ganancias[indice_canal]
                                         + desplazamientos[indice_canal])
    return _acotar_rango(resultado)



VARIANTES_FOTOMETRICAS = ("luminancia", "balance_rgb", "mixta")
 
VARIANTES_SORTEABLES = ("luminancia", "balance_rgb")


def sortear_variante_fotometrica(semilla, opciones=VARIANTES_SORTEABLES):
    generador = np.random.default_rng([int(semilla), 20252])
    return str(generador.choice(list(opciones)))

class DegradacionFotometricaPorPieza:
    """
    Aplica a cada pieza una transformacion puntual distinta.
    Se usa como `degradacion_por_pieza`.
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

    def _sortear_luminancia(self, generador):
        return {
            "gamma": float(generador.uniform(*self.rango_gamma)),
            "contraste": float(generador.uniform(*self.rango_contraste)),
            "brillo": float(generador.uniform(*self.rango_brillo)),
        }
 
    def _sortear_balance_rgb(self, generador):
        return {
            "ganancias": [float(generador.uniform(*self.rango_ganancia)) for _ in range(3)],
            "desplazamientos": [float(generador.uniform(*self.rango_desplazamiento))
                                for _ in range(3)],

    def __call__(self, pieza, indice, generador):
        registro = {"variante": self.variante}
        resultado = pieza
 
        if self.variante in ("luminancia", "mixta"):
            parametros = self._sortear_luminancia(generador)
            resultado = alterar_luminancia(resultado, **parametros)
            registro.update({k: round(v, 3) for k, v in parametros.items()})
 
        if self.variante in ("balance_rgb", "mixta"):
            parametros = self._sortear_balance_rgb(generador)
            resultado = alterar_balance_rgb(resultado, **parametros)
            registro.update({k: [round(x, 3) for x in v] for k, v in parametros.items()})
 
        self.parametros_por_pieza[int(indice)] = registro
        return resultado
 
    def reiniciar(self):
        self.parametros_por_pieza = {}
