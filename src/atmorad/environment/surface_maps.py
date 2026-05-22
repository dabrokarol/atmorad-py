from abc import ABC, abstractmethod

import numpy as np

from atmorad.constants import X, Y

from .registry import register_surface_map


class BaseSurfaceMap(ABC):
    @abstractmethod
    def get_material_ids(self, pos: np.ndarray) -> np.ndarray:
        """Returns an array of material IDs corresponding to photon coordinates."""
        pass


@register_surface_map("uniform", ["material"])
class UniformMap(BaseSurfaceMap):
    def get_material_ids(self, pos: np.ndarray):
        return np.zeros_like(pos[X], dtype=int)


@register_surface_map("split-half-x", ["material_left", "material_right"])
class SplitHalfXMap(BaseSurfaceMap):
    def get_material_ids(self, pos: np.ndarray):
        return np.where(pos[X] < 0, 0, 1)


@register_surface_map("circle", ["material_in", "material_out"])
class CircleMap(BaseSurfaceMap):
    def __init__(self, radius_km: float):
        self.radius_sq = radius_km**2

    def get_material_ids(self, pos: np.ndarray):
        return np.where((pos[X] ** 2 + pos[Y] ** 2) < self.radius_sq, 0, 1)


@register_surface_map("checkerboard", ["material_a", "material_b"])
class CheckerboardMap(BaseSurfaceMap):
    def __init__(self, tile_size_km: float):
        self.tile_size = tile_size_km

    def get_material_ids(self, pos: np.ndarray):
        x = np.mod(pos[X], self.tile_size)
        y = np.mod(pos[Y], self.tile_size)
        half = self.tile_size / 2.0
        return np.where(((x < half) & (y < half)) | ((x >= half) & (y >= half)), 0, 1)


@register_surface_map("grid", ["materials"])
class GridMap(BaseSurfaceMap):
    """
    Args:
        ground_ids_matrix: 2D array of shape (X, Y) containing integer material IDs for each cell.
        cell_size_km: Size of each grid cell in kilometers.
        periodic: Whether the grid is periodic. If non-periodic, out-of-bounds coordinates
                  will be clamped to the edge cells.
    """

    def __init__(
        self,
        ground_ids_matrix: np.ndarray,
        cell_size_km: float = 1.0,
        periodic: bool = True,
    ):
        self.matrix = np.asarray(ground_ids_matrix, dtype=int)
        self.cell_size = cell_size_km
        self.periodic = periodic
        self.n_x, self.n_y = self.matrix.shape

    def get_material_ids(self, pos: np.ndarray) -> np.ndarray:
        grid_x = pos[X] / self.cell_size
        grid_y = pos[Y] / self.cell_size

        if self.periodic:
            idx_x = np.mod(np.floor(grid_x), self.n_x).astype(int)
            idx_y = np.mod(np.floor(grid_y), self.n_y).astype(int)
        else:
            idx_x = np.clip(np.floor(grid_x), 0, self.n_x - 1).astype(int)
            idx_y = np.clip(np.floor(grid_y), 0, self.n_y - 1).astype(int)

        return self.matrix[idx_x, idx_y]
