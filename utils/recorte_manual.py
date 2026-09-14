"""
Herramienta axuliar para elegir a mano el recorte de cada fotografia original
No forma parte del pipeline de pipeline general: se usa una sola vez por imagen, de forma interactiva,y su unica salida es una entrada en config/recortes.json.
"""

#Librerias
import json
from pathlib import Path
from skimage import transform
from core.preparacion_imagenes import LADO_BASE, cargar_imagen,extraer_parche_cuadrado
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from ipywidgets import interact, IntSlider, Button, Output, VBox
from IPython.display import display


__all__ = ["RUTA_CONFIG","previsualizar_recorte","selector_interactivo","registrar_configuracion_recorte","leer_config"]

#Variables globales
RUTA_CONFIG = Path("config/recortes.json") 
LADO_MINIATURA = 600 #previsulizaicon del recorte

def generar_miniatura(matriz_imagen, lado_maximo = LADO_MINIATURA):
    """
    Genera una versión reducida de la imagen manteniendo la relación de aspecto.
    Devuelve la miniatura y el factor de escala aplicado.
    """

    alto_px, ancho_px = matriz_imagen.shape[:2]
    factor_escala = max(alto_px, ancho_px) / lado_maximo
    
    forma_miniatura = (int(alto_px / factor_escala), int(ancho_px / factor_escala))
    miniatura = transform.resize(matriz_imagen, forma_miniatura, anti_aliasing=True)
    
    return miniatura, factor_escala

def dibujar_imagen_con_recorte(ax, miniatura, factor_escala, ancho_original, alto_original, longitud_lado_px, esquina_x, esquina_y):
    """Configura el panel dibujando la miniatura con el recuadro del recorte."""

    #muestra recuadro
    ax.imshow(miniatura)
    ax.add_patch(Rectangle((esquina_x / factor_escala, esquina_y / factor_escala),
        longitud_lado_px / factor_escala, 
        longitud_lado_px / factor_escala,
        linewidth=2.5, 
        edgecolor="red", 
        facecolor="none",
    ))
    ax.set_title(f"Original {ancho_original}x{alto_original}px | Recorte en ({esquina_x}, {esquina_y})")
    ax.axis("off")

def previsualizar_recorte(matriz_imagen, longitud_lado_px, esquina_x, esquina_y):
    """
    Muestra dos paneles: la foto completa con el recuadro dibujado encima, y el recorte resultante
    """
    alto_px, ancho_px = matriz_imagen.shape[:2]

    miniatura, factor_escala = generar_miniatura(matriz_imagen)

    recorte = extraer_parche_cuadrado(matriz_imagen, longitud_lado_px, esquina_x, esquina_y)

    figura, (panel_izq, panel_der) = plt.subplots(1, 2, figsize=(13, 6))

    dibujar_imagen_con_recorte(panel_izq, miniatura, factor_escala, ancho_px, alto_px, longitud_lado_px, esquina_x, esquina_y)

    panel_der.imshow(recorte)
    panel_der.set_title(f"Recorte {longitud_lado_px}x{longitud_lado_px}px")
    panel_der.axis("off")

    plt.tight_layout()
    plt.show()

def leer_config(ruta_config=RUTA_CONFIG):
    """
    Lee un archivo de configuración JSON. 
    Devuelve un diccionario vacío si el archivo no existe.
    """
    ruta_config = Path(ruta_config)
    if not ruta_config.exists():
        return {}
    return json.loads(ruta_config.read_text(encoding="utf-8"))

def guardar_json_config(configuracion, ruta_config):
    """
    Función auxiliar que asegura la creación del directorio y escribe el JSON
    """
    ruta_config.parent.mkdir(parents=True, exist_ok=True)
    ruta_config.write_text(json.dumps(configuracion, indent=2, ensure_ascii=False),encoding="utf-8")

def registrar_configuracion_recorte(identificador_imagen, ruta_relativa_original, longitud_lado_px, esquina_x, esquina_y,
                      lado_objetivo_px = LADO_BASE, ruta_destino_png = None, ruta_archivo_config = RUTA_CONFIG, verbose =True):
    """
    Registra los parametros de corte e inspeccion visual en el archivo de configuracion JSON
    """
    path_config = Path(ruta_archivo_config)
    diccionario_configuraciones = leer_config(path_config)

    ruta_salida_efectiva = (str(ruta_destino_png) if ruta_destino_png else f"imagenes/base/{identificador_imagen}.png")

    # Construcción de la nueva entrada
    registro_recorte = {
        "ruta_imagen_original": str(ruta_relativa_original),
        "ruta_imagen_procesada": ruta_salida_efectiva,
        "coordenadas_corte_origen": {
            "x_px": int(esquina_x),
            "y_px": int(esquina_y),
            "ancho_px": int(longitud_lado_px),
            "alto_px": int(longitud_lado_px)
        },
        "transformacion_destino": {
            "ancho_final_px": int(lado_objetivo_px),
            "alto_final_px": int(lado_objetivo_px)
        }
    }

    diccionario_configuraciones[identificador_imagen] = registro_recorte
 
    guardar_json_config(diccionario_configuraciones, path_config)
 
    if verbose:
        corte = registro_recorte["coordenadas_corte_origen"]
        dest = registro_recorte["transformacion_destino"]
        print(f"Guardado '{identificador_imagen}' en {path_config}:")
        print(f"origen : {registro_recorte['ruta_imagen_original']}  (relativa a la raiz de originales)")
        print(f"recorte: {corte['ancho_px']}x{corte['alto_px']}px en ({corte['x_px']}, {corte['y_px']})")
        print(f"salida : {registro_recorte['ruta_imagen_procesada']} ({dest['ancho_final_px']}x{dest['alto_final_px']}px)")
 
    return registro_recorte


def selector_interactivo(ruta_relativa_origen, identificador_imagen, raiz_originales="", longitud_lado_px=None, lado_objetivo_px=LADO_BASE, 
                         ruta_config=RUTA_CONFIG):
    """
    Son los controles interactivos (ipywidgets) para ajustar el recorte y guardarlo en el JSON.
    """
    ruta_completa = Path(raiz_originales) / ruta_relativa_origen
 
    if not ruta_completa.exists():
        raise FileNotFoundError(
            f"No se encontro la imagen en '{ruta_completa}'. "
            f"Revisar 'raiz_originales' ({raiz_originales}) y 'ruta_relativa_origen' ({ruta_relativa_origen})."
        )

    matriz_imagen = cargar_imagen(ruta_completa)
    alto_px, ancho_px = matriz_imagen.shape[:2]

    if longitud_lado_px is None:
        longitud_lado_px = min(alto_px, ancho_px)

    #Centramos por defecto
    x_inicial = (ancho_px - longitud_lado_px) // 2
    y_inicial = (alto_px - longitud_lado_px) // 2

    estado = {"x": x_inicial, "y": y_inicial}

    def _actualizar(x, y):
        estado["x"], estado["y"] = x, y
        previsualizar_recorte(matriz_imagen, longitud_lado_px, x, y)

    interact(_actualizar,
        x=IntSlider(value=estado["x"], min=0, max=ancho_px - longitud_lado_px, step=10, description="x", continuous_update=False),
        y=IntSlider(value=estado["y"], min=0, max=alto_px - longitud_lado_px, step=10, description="y", continuous_update=False),
    )

    boton_guardar = Button(description="Guardar Configuración", button_style="success")
    salida = Output()

    def _al_hacer_click(_):
        with salida:
            salida.clear_output()
            registrar_configuracion_recorte(
                identificador_imagen=identificador_imagen,
                ruta_relativa_original=ruta_relativa_origen,
                longitud_lado_px=longitud_lado_px,
                esquina_x=estado["x"],
                esquina_y=estado["y"],
                lado_objetivo_px=lado_objetivo_px,
                ruta_archivo_config=ruta_config
            )

    boton_guardar.on_click(_al_hacer_click)
    display(VBox([boton_guardar, salida]))