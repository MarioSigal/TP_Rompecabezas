from typing import Callable, Dict, List, Optional, Tuple, Any, Union
import numpy as np
import numpy.typing as npt

BoolArray = npt.NDArray[np.bool_]

ActualizacionDePieza = Tuple[np.ndarray, Optional[BoolArray]]

DegradadorPorPieza = Callable[[np.ndarray, int, np.random.Generator, Optional[BoolArray]], ActualizacionDePieza]
DegradadorGlobal = Callable[[np.ndarray, np.random.Generator], np.ndarray]
RangoFlotante = tuple[float,float]