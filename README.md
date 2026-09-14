# 🧩 TP Rompecabezas — Procesamiento Digital de Imágenes (PDI)

Infraestructura integral para la generación, evaluación y resolución automatizada de rompecabezas con degradaciones progresivas de procesamiento digital de imágenes.

---

## 📚 1. Filosofía del Trabajo Práctico

El pipeline del trabajo práctico se divide deliberadamente en dos partes:

| Componente | Responsable | Qué resuelve |
|---|---|---|
| **Limpieza y Métricas** | **El Alumno** | Identificar el tipo de ruido o degradación, diseñar filtros espaciales/frecuenciales, y formular la función de costo/afinidad entre pares de piezas. |
| **Búsqueda del Armado** | **La Cátedra** | Dada la matriz de afinidades o la función de compatibilidad suministrada por el alumno, resuelve la colocación espacial óptima en la grilla. |

El reconstructor se entrega listo y **no requiere modificaciones**. Los estudiantes pueden enchufar **cualquier función de compatibilidad personalizada** con la firma estándar:

```python
def mi_compatibilidad(pieza_a: np.ndarray, pieza_b: np.ndarray, relacion: str) -> float:
    """
    relacion: 'horizontal' (B a la derecha de A) o 'vertical' (B abajo de A).
    Retorna un valor float de costo/disimilitud (menor valor = mayor compatibilidad).
    """
    ...
```

---

## 🎯 2. Los 5 Niveles del TP

El trabajo práctico está estructurado en **5 niveles pedagógicos progresivos**:

### 🔹 Nivel 1: Ruido Espacial Aditivo e Impulsivo (6 Variantes Oficiales)
- **Geometría:** Piezas cuadradas regulares.
- **Degradación:** La cátedra provee 6 variantes de ruido mixto:
  1. `Gaussiano (sigma=0.08) + sal y pimienta 3%`
  2. `Sal y pimienta 8% + uniforme [-0.06, 0.06]`
  3. `Gaussiano (sigma=0.05) + SOLO SAL 6%`
  4. `Rayleigh (b=0.020) + SOLO PIMIENTA 7%`
  5. `Uniforme [-0.10, 0.10] + sal y pimienta 4%`
  6. `Gaussiano (sigma=0.07) + impulsivo asimetrico (sal 4.5%, pimienta 1%)`
- **Desafío:** Identificar el ruido en el histograma y aplicar **filtros espaciales óptimos** (filtro de mediana, media, alfa-recortada, mínimo para sal o máximo para pimienta) para restaurar los bordes antes de medir afinidad.

### 🔹 Nivel 2: Variaciones Fotométricas por Pieza y Ecualización en Luminancia
- **Geometría:** Piezas cuadradas regulares.
- **Degradación:** Cada pieza sufre una alteración individual de gamma ($\gamma \in [0.6, 1.7]$), brillo, contraste y saturación en HSV/YCrCb.
- **Desafío:** **Ecualizar el canal de luminancia** (por ejemplo canal $Y$ en YCrCb o $V$ en HSV con `cv2.equalizeHist` tal como se vio en clase) o proponer métricas de continuidad para homogenizar los niveles de intensidad respetando el color.

### 🔹 Nivel 3: Filtrado en Frecuencia (Fourier 2D)
- **Geometría:** Piezas cuadradas regulares.
- **Degradación:** Ruido armónico periódico (interferencia sinusoidal / muaré) que genera picos espectrales discretos de alta energía en el dominio de Fourier.
- **Desafío:** Calcular la Transformada 2D de Fourier (`np.fft.fft2`), centrar con `fftshift`, diseñar e implementar **filtros de muesca (Notch Filters)** que anulen las frecuencias de interferencia y antitransformar.

### 🔹 Nivel 4: Geometría de Encastres Curvos (Jigsaw)
- **Geometría:** Siluetas poligonales recortadas sobre **fondo negro puro `(0, 0, 0)`**.
- **Degradación:** Encastres analíticos de tipo **Saliente (+1)** (pestaña), **Entrante (-1)** (muesca/hendidura) y **Plano (0)** perimetral.
- **Desafío:** Utilizar análisis morfológico de contornos (`core.detector_forma`) y formular una métrica híbrida que verifique complementariedad de forma y continuidad cromática:
  $$\text{costo}(A, B) = \text{costo\_forma}(A, B) + \lambda \cdot \text{costo\_color}(A, B)$$

### 🔹 Nivel 5: Rotaciones y Modulación Periódica Horizontal (Deskewing)
- **Geometría:** Piezas rotadas (múltiplos de 90° e inclinaciones continuas leves de $\pm 10^\circ$).
- **Degradación:** La imagen original posee un filtro de rayas horizontales periódicas ($I(y, x) \cdot (1 - \alpha \sin^2(\pi y / T))$).
- **Desafío:** Detectar el ángulo de rotación de cada pieza buscando el pico espectral en el dominio de Fourier 2D, **enderezar (deskewing)** cada pieza a su orientación horizontal ($0^\circ$) y resolver el ensamble.

---

## 📁 3. Estructura de la Carpeta

```
TP_ROMPECABEZAS_CURSO/
├── core/
│   ├── __init__.py
│   ├── preparacion_imagenes.py    # Carga, normalización float64 [0, 1] RGB, recorte base 1024x1024
│   ├── degradaciones.py           # Ruidos espaciales, fotométricos por pieza y periódicos en frecuencia
│   ├── crear_rompecabezas.py      # Generador unificado crear_rompecabezas_nivel(img, nivel=1..5)
│   ├── bordes.py                  # Extracción de bandas perimetrales y matrices de afinidad
│   ├── geometria_jigsaw.py        # Curvas analíticas y extracción de siluetas de piezas con encastre
│   ├── detector_forma.py          # Binarización, segmentación de contornos y matching de curvas
│   ├── analizador_rotacion.py     # Modulación horizontal, estimación angular en FFT 2D y deskewing
│   ├── metricas.py                # Top-1, MRR, Vecindad, Directa, PSNR y SSIM
│   └── reconstructor.py           # Reconstructor universal sobre tablero expandible (Best-First + Backtracking)
├── catedra/
│   ├── __init__.py
│   ├── filtros_referencia.py      # Implementaciones docentes de filtros espaciales, CLAHE y Notch Fourier
│   └── soluciones_referencia.py   # Pipelines de resolución de referencia para auditoría y evaluación
├── utils/
│   ├── __init__.py
│   ├── visualizacion.py           # Plots para notebooks (piezas, espectros FFT, comparativas, grillas)
│   └── animacion.py               # Generación de GIFs animados paso a paso
├── imagenes/
│   └── base/                      # Imágenes de prueba oficiales (paisaje.png, mandril.png)
├── tests/
│   └── test_5_niveles.py          # Suite de pruebas automatizadas de integración
├── TP_Rompecabezas_Colab.ipynb    # Notebook interactivo oficial para los estudiantes
├── requirements.txt               # Dependencias del proyecto
└── README.md                      # Documentación principal
```

---

## 🚀 4. Uso Rápido

### Generar un rompecabezas de cualquier nivel:
```python
from core import cargar_imagen, crear_rompecabezas_nivel

# Cargar imagen base en float64 [0.0, 1.0] RGB
img = cargar_imagen('imagenes/base/paisaje.png')

# Crear rompecabezas indicando el nivel deseado (1 al 5)
puzzle = crear_rompecabezas_nivel(img, nivel=3, filas=3, columnas=3, semilla=42)
print(f"Piezas generadas: {puzzle.cantidad_piezas}")
```

### Resolver con una métrica personalizada:
```python
from core import reconstruir_rompecabezas, generar_reporte_completo, imprimir_reporte

def mi_metrica(pieza_a, pieza_b, relacion):
    # Definida por el alumno
    ...
    return costo

# El reconstructor ejecuta automáticamente con la función del alumno:
grilla = reconstruir_rompecabezas(
    piezas=puzzle.piezas,
    cantidad_filas=puzzle.cantidad_filas,
    cantidad_columnas=puzzle.cantidad_columnas,
    funcion_compatibilidad=mi_metrica
)

reporte = generar_reporte_completo(puzzle, grilla_propuesta=grilla)
imprimir_reporte(reporte, titulo="Resultado del Alumno")
```

### Correr las pruebas de integración:
```bash
python tests/test_5_niveles.py
```
