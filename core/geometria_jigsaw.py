"""
Módulo de Geometría de Rompecabezas con Encastres (Tabs & Blanks / Saliente y Entrante).
Genera curvas bezier y analíticas continuas entre fichas adyacentes tipo Jigsaw.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import cv2


#Tabla finita de generacion de bordes
TABLA_BORDES_DISCRETOS = {
    "perfiles": ("gaussiano", "semicircular"),
    "posiciones": (0.35, 0.425, 0.50, 0.575, 0.65),
    "profundidades": (0.16, 0.20, 0.24),
    "anchos": (0.24, 0.28, 0.32, 0.36),            
    "ancho": 0.30,
}

__all__ = ["generate_tab_curve", "JigsawGridGeometry", "TABLA_BORDES_DISCRETOS"]



def _calcular_bulb(prof: str, x_norm: float) -> float:
    #Desplazamiento perpendicular normalizado segun el perfil del encastre
    if prof in ("circular", "semicircular"):
        return float(np.sqrt(max(0.0, 1.0 - x_norm ** 2)))
    elif prof in ("wide", "random"):
        return float(np.cos(x_norm * np.pi / 2.0) ** 1.3)
    elif prof in ("gaussiano", "gaussian"):
        return float(np.exp(-2.5 * (x_norm ** 2)))
    else:
        return float(np.cos(x_norm * np.pi / 2.0) ** 1.6)



def generate_tab_curve(
    p_start: Tuple[float, float],
    p_end: Tuple[float, float],
    tab_type: int,
    profile_type: str = "standard", 
    num_points: int = 60,
    tab_depth_ratio: float = 0.20,
    tab_width_ratio: float = 0.32,
) -> np.ndarray:
    """
    Genera los puntos (x, y) de la curva de un borde con encastre analítico realista.
    """
    p0 = np.array(p_start, dtype=np.float32)
    p1 = np.array(p_end, dtype=np.float32)

    vec = p1 - p0
    length = float(np.linalg.norm(vec))
    if length == 0 or tab_type == 0:
        t = np.linspace(0, 1, num_points)[:, None]
        return p0 + t * vec

    u = vec / length  # Vector unitario tangente
    # Vector normal unitario (90° horario del vector tangente)
    n = np.array([u[1], -u[0]], dtype=np.float32)

    depth = length * tab_depth_ratio * float(tab_type)
    width = length * tab_width_ratio
    center = 0.5 * length

    t_vals = np.linspace(0, 1, num_points)
    curve_points = []

    for t in t_vals:
        s = t * length
        dist_from_center = (s - center) / (width / 2.0)

        if abs(dist_from_center) <= 1.0:
            x_norm = float(dist_from_center)
            if profile_type == "circular":
                bulb = np.sqrt(max(0.0, 1.0 - x_norm ** 2))
            elif profile_type in ("wide", "random"):
                bulb = np.cos(x_norm * np.pi / 2.0) ** 1.3
            else:
                bulb = np.cos(x_norm * np.pi / 2.0) ** 1.6
            offset = depth * bulb
        else:
            offset = 0.0

        pt = p0 + s * u + offset * n
        curve_points.append(pt)

    return np.array(curve_points, dtype=np.float32)


class JigsawGridGeometry:
    """
    Gestiona la coherencia global de los encastres de una grilla R x C.
    Garantiza que si el borde Este de la pieza (r, c) es Saliente (+1),
    el borde Oeste de la pieza (r, c+1) sea el encastre Entrante (-1) complementario exacto.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        image_h: int,
        image_w: int,
        seed: Optional[int] = 42,
        profile_types: Optional[List[str]] = None,
        discrete: bool = False,
    ):
        self.rows = rows
        self.cols = cols
        self.img_h = image_h
        self.img_w = image_w
 
        self.tile_h = image_h // rows
        self.tile_w = image_w // cols
        self.discrete = discrete
 
        rng = np.random.default_rng(seed)
 
        self.horiz_tabs = rng.choice([1, -1], size=(rows, cols - 1))
 
        self.vert_tabs = rng.choice([1, -1], size=(rows - 1, cols))
 
        if discrete:
            perfiles = TABLA_BORDES_DISCRETOS["perfiles"]
            posiciones = TABLA_BORDES_DISCRETOS["posiciones"]
            profundidades = TABLA_BORDES_DISCRETOS["profundidades"]
            anchos = TABLA_BORDES_DISCRETOS["anchos"]
 
            self.allowed_profiles = list(perfiles)
            self.horiz_profile_types = rng.choice(perfiles, size=(rows, cols - 1))
            self.vert_profile_types = rng.choice(perfiles, size=(rows - 1, cols))
 
            self.horiz_centers = rng.choice(posiciones, size=(rows, cols - 1))
            self.vert_centers = rng.choice(posiciones, size=(rows - 1, cols))
            self.horiz_depths = rng.choice(profundidades, size=(rows, cols - 1))
            self.vert_depths = rng.choice(profundidades, size=(rows - 1, cols))
            self.horiz_widths = rng.choice(anchos, size=(rows, cols - 1))
            self.vert_widths = rng.choice(anchos, size=(rows - 1, cols))
        else:
            self.allowed_profiles = profile_types if profile_types is not None else ["standard", "circular", "wide"]
            self.horiz_profile_types = rng.choice(self.allowed_profiles, size=(rows, cols - 1))
            self.vert_profile_types = rng.choice(self.allowed_profiles, size=(rows - 1, cols))
 
            self.horiz_centers = rng.uniform(0.40, 0.60, size=(rows, cols - 1))
            self.vert_centers = rng.uniform(0.40, 0.60, size=(rows - 1, cols))
            self.horiz_depths = rng.uniform(0.18, 0.22, size=(rows, cols - 1))
            self.vert_depths = rng.uniform(0.18, 0.22, size=(rows - 1, cols))
            self.horiz_widths = rng.uniform(0.28, 0.34, size=(rows, cols - 1))
            self.vert_widths = rng.uniform(0.28, 0.34, size=(rows - 1, cols))
 
        self._seams_v = {} 
        self._seams_h = {}
        self._compute_canonical_seams(num_pts=60)


    def _compute_canonical_seams(self, num_pts: int = 60) -> None:
        """Calcula una única vez cada curva de unión interior compartida."""
        # 1. Costuras horizontales: entre fila r y r+1
        for r in range(self.rows - 1):
            for c in range(self.cols):
                p0 = np.array([c * self.tile_w, (r + 1) * self.tile_h], dtype=np.float32)
                p1 = np.array([(c + 1) * self.tile_w, (r + 1) * self.tile_h], dtype=np.float32)
                prof = str(self.vert_profile_types[r, c])
                tab_dir = int(self.vert_tabs[r, c])
 
                u = (p1 - p0) / float(self.tile_w)
                n = np.array([0.0, 1.0], dtype=np.float32) if tab_dir == 1 else np.array([0.0, -1.0], dtype=np.float32)
 
                length = float(self.tile_w)
                depth = length * float(self.vert_depths[r, c])
                width = length * float(self.vert_widths[r, c])
                center = float(self.vert_centers[r, c]) * length
 
                pts = []
                for t in np.linspace(0.0, 1.0, num_pts):
                    s = t * length
                    dist = (s - center) / (width / 2.0)
                    if abs(dist) <= 1.0:
                        x_norm = float(dist)
                        bulb = _calcular_bulb(prof, x_norm)
                    else:
                        bulb = 0.0
                    pts.append(p0 + s * u + (depth * bulb) * n)
                self._seams_v[(r, c)] = np.array(pts, dtype=np.float32)
 
        # 2. Costuras verticales: entre columna c y c+1
        for r in range(self.rows):
            for c in range(self.cols - 1):
                p0 = np.array([(c + 1) * self.tile_w, r * self.tile_h], dtype=np.float32)
                p1 = np.array([(c + 1) * self.tile_w, (r + 1) * self.tile_h], dtype=np.float32)
                prof = str(self.horiz_profile_types[r, c])
                tab_dir = int(self.horiz_tabs[r, c])
 
                u = (p1 - p0) / float(self.tile_h)
                n = np.array([1.0, 0.0], dtype=np.float32) if tab_dir == 1 else np.array([-1.0, 0.0], dtype=np.float32)
 
                length = float(self.tile_h)
                depth = length * float(self.horiz_depths[r, c])
                width = length * float(self.horiz_widths[r, c])
                center = float(self.horiz_centers[r, c]) * length
 
                pts = []
                for t in np.linspace(0.0, 1.0, num_pts):
                    s = t * length
                    dist = (s - center) / (width / 2.0)
                    if abs(dist) <= 1.0:
                        x_norm = float(dist)
                        bulb = _calcular_bulb(prof, x_norm)
                    else:
                        bulb = 0.0
                    pts.append(p0 + s * u + (depth * bulb) * n)
                self._seams_h[(r, c)] = np.array(pts, dtype=np.float32)

    def get_piece_edge_types(self, r: int, c: int) -> Dict[str, str]:
        """Devuelve 'PLANO', 'SALIENTE', 'ENTRANTE' para cada lado N, S, E, W de la pieza (r, c)."""
        type_n = "PLANO" if r == 0 else ("ENTRANTE" if self.vert_tabs[r - 1, c] == 1 else "SALIENTE")
        type_s = "PLANO" if r == self.rows - 1 else ("SALIENTE" if self.vert_tabs[r, c] == 1 else "ENTRANTE")
        type_w = "PLANO" if c == 0 else ("ENTRANTE" if self.horiz_tabs[r, c - 1] == 1 else "SALIENTE")
        type_e = "PLANO" if c == self.cols - 1 else ("SALIENTE" if self.horiz_tabs[r, c] == 1 else "ENTRANTE")
        return {"NORTE": type_n, "SUR": type_s, "OESTE": type_w, "ESTE": type_e}

    def get_piece_edge_curves(self, r: int, c: int) -> Dict[str, str]:
        """Devuelve el perfil ('none', 'standard', 'circular', 'wide') para cada lado."""
        prof_n = "none" if r == 0 else str(self.vert_profile_types[r - 1, c])
        prof_s = "none" if r == self.rows - 1 else str(self.vert_profile_types[r, c])
        prof_w = "none" if c == 0 else str(self.horiz_profile_types[r, c - 1])
        prof_e = "none" if c == self.cols - 1 else str(self.horiz_profile_types[r, c])
        return {"NORTE": prof_n, "SUR": prof_s, "OESTE": prof_w, "ESTE": prof_e}

    def get_piece_edges(self, r: int, c: int, num_pts: int = 60) -> Dict[str, np.ndarray]:
        """Retorna las 4 curvas del contorno de la pieza (r, c) orientadas en sentido horario."""
        if r == 0:
            curve_n = np.array([[x, 0.0] for x in np.linspace(c * self.tile_w, (c + 1) * self.tile_w, num_pts)], dtype=np.float32)
        else:
            curve_n = self._seams_v[(r - 1, c)].copy()
 
        if c == self.cols - 1:
            curve_e = np.array([[(c + 1) * self.tile_w, y] for y in np.linspace(r * self.tile_h, (r + 1) * self.tile_h, num_pts)], dtype=np.float32)
        else:
            curve_e = self._seams_h[(r, c)].copy()
 
        if r == self.rows - 1:
            curve_s = np.array([[x, (r + 1) * self.tile_h] for x in np.linspace((c + 1) * self.tile_w, c * self.tile_w, num_pts)], dtype=np.float32)
        else:
            curve_s = self._seams_v[(r, c)][::-1].copy()
 
        if c == 0:
            curve_w = np.array([[0.0, y] for y in np.linspace((r + 1) * self.tile_h, r * self.tile_h, num_pts)], dtype=np.float32)
        else:
            curve_w = self._seams_h[(r, c - 1)][::-1].copy()
 
        return {"NORTE": curve_n, "ESTE": curve_e, "SUR": curve_s, "OESTE": curve_w}

    def get_piece_polygon(self, r: int, c: int, num_pts_per_edge: int = 60) -> np.ndarray:
        """Retorna el polígono 2D cerrado en sentido horario."""
        edges = self.get_piece_edges(r, c, num_pts=num_pts_per_edge)
        return np.vstack([edges["NORTE"], edges["ESTE"], edges["SUR"], edges["OESTE"]])

    def extract_piece_image(
        self,
        full_image: np.ndarray,
        r: int,
        c: int,
        padding: int = 35,
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        """
        Recorta la pieza correspondiente a (r, c) con su forma real de encastre sobre fondo negro (0, 0, 0).
        Soporta arrays float64 [0, 1] y uint8.

        Returns:
            - piece_img: Imagen recortada centrada sobre fondo negro.
            - piece_mask: Máscara binaria (uint8: 255 pieza, 0 fondo).
            - (offset_x, offset_y): Coordenadas globales de la esquina sup-izq del parche.
        """
        poly = self.get_piece_polygon(r, c)

        pad_x = int(np.ceil(self.tile_w * 0.25)) + padding
        pad_y = int(np.ceil(self.tile_h * 0.25)) + padding

        crop_w = self.tile_w + 2 * pad_x
        crop_h = self.tile_h + 2 * pad_y

        min_x = c * self.tile_w - pad_x
        max_x = min_x + crop_w
        min_y = r * self.tile_h - pad_y
        max_y = min_y + crop_h

        canales = full_image.shape[2] if full_image.ndim == 3 else 1
        piece_img = np.zeros((crop_h, crop_w, canales), dtype=full_image.dtype)
        mask = np.zeros((crop_h, crop_w), dtype=np.uint8)

        src_x0 = max(0, min_x)
        src_x1 = min(self.img_w, max_x)
        src_y0 = max(0, min_y)
        src_y1 = min(self.img_h, max_y)

        dst_x0 = src_x0 - min_x
        dst_x1 = dst_x0 + (src_x1 - src_x0)
        dst_y0 = src_y0 - min_y
        dst_y1 = dst_y0 + (src_y1 - src_y0)

        if src_x1 > src_x0 and src_y1 > src_y0:
            patch = full_image[src_y0:src_y1, src_x0:src_x1]
            if issubclass(full_image.dtype.type, np.floating):
                # Asegurar valor mínimo pequeño para no confundir con fondo negro puro
                piece_img[dst_y0:dst_y1, dst_x0:dst_x1] = np.maximum(patch, 1e-4)
            else:
                piece_img[dst_y0:dst_y1, dst_x0:dst_x1] = np.maximum(patch, 1)

        local_poly = poly.copy()
        local_poly[:, 0] -= min_x
        local_poly[:, 1] -= min_y
        local_poly_int = np.round(local_poly).astype(np.int32)

        cv2.fillPoly(mask, [local_poly_int], 255)

        if piece_img.ndim == 3:
            for ch in range(canales):
                piece_img[:, :, ch] = np.where(mask > 0, piece_img[:, :, ch], 0)
        else:
            piece_img = np.where(mask > 0, piece_img, 0)

        return piece_img, mask, (min_x, min_y)
