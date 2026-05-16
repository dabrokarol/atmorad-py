import numpy as np

from typing import Sequence
from abc import ABC, abstractmethod
from dataclasses import dataclass

from atmorad.engine.batch import PhotonBatch
from atmorad.physics.reflection import SurfaceReflection
from atmorad.constants import EPSILON, X, Y, Z

@dataclass
class SurfaceMaterial:
    albedo: float
    reflection: SurfaceReflection
    def __post_init__(self):
        if not 0.0 <= self.albedo <= 1.0:
            raise ValueError(f'Albedo should be non-negative, got {self.albedo}.')

class SurfaceMap(ABC):
    @abstractmethod
    def get_material_ids(self, pos: np.ndarray) -> np.ndarray:
        ...

class ProceduralMap(SurfaceMap):
    def __init__(self, procedure):
        self.procedure = procedure

    def get_material_ids(self, pos):
        return self.procedure(pos)
    
    #example map functions:
    @staticmethod
    def uniform_ground(pos):
        return np.zeros_like(pos[X]).astype(int)
    
    @staticmethod
    def split_half_x(pos):
        return np.where(pos[X] < 0, 0, 1)
    
    @staticmethod
    def circle(pos, radius):
        return np.where((pos[X]**2 + pos[Y]**2) < radius**2, 0, 1)
    
    @staticmethod
    def checkerboard(pos, tile_size):
        x = np.mod(pos[X], tile_size)
        y = np.mod(pos[Y], tile_size)
        return np.where(((x<tile_size/2) & (y<tile_size/2)) | ((x >= tile_size/2) & (y >= tile_size/2)), 0, 1)

class GridMap(SurfaceMap):    
    """
    Args:
        grounds_ids_matrix: 2D array of shape (X, Y) containing integer material IDs for each cell.
        cell_size_km: Size of each grid cell in kilometers.
        periodic: Whether the grid is periodic. If non-periodic, out-of-bounds coordinates will be clamped to the edge cells.
    """
    def __init__(self, ground_ids_matrix: np.ndarray, cell_size_km: float = 1.0, periodic: bool = True):
        self.matrix = np.asarray(ground_ids_matrix, dtype=int)
        self.cell_size = cell_size_km
        self.periodic = periodic
        
        self.n_x = self.matrix.shape[0]
        self.n_y = self.matrix.shape[1]
        
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

from abc import ABC, abstractmethod

class Surface(ABC):
    """Abstract Base Class for all terrain types."""
    
    @abstractmethod
    def process_reflection(self, batch: PhotonBatch, surface_mask: np.ndarray, random_samples: np.ndarray) -> tuple[PhotonBatch, np.ndarray]:
        ...
        
    @abstractmethod
    def is_below_ground(self, pos: np.ndarray) -> np.ndarray:
        ...
        
    @abstractmethod
    def adjust_surface_boundary(self, batch: PhotonBatch) -> PhotonBatch:
        ...

class FlatSurface(Surface):
    def __init__(self, ground_map: SurfaceMap, ground_types: Sequence[SurfaceMaterial]):
        self.ground_map = ground_map
        self.ground_types = np.array(ground_types)
        self.albedos = np.array([material.albedo for material in self.ground_types])
        self.reflections = [material.reflection for material in self.ground_types]
    
    def process_reflection(self, batch: PhotonBatch, surface_mask: np.ndarray, random_samples: np.ndarray):
        pos_hit = batch.pos[:, surface_mask]
        material_ids = self.ground_map.get_material_ids(pos_hit)

        rand_albedo = random_samples[0, surface_mask]
        survived_albedo = rand_albedo < self.albedos[material_ids]
        
        to_reflect = np.zeros(batch.active_count, dtype=bool)
        to_reflect[surface_mask] = survived_albedo
        
        if np.any(survived_albedo):
            survivor_materials = material_ids[survived_albedo]
            survivor_dirs = batch.direction[:, to_reflect]
            survivor_r1 = random_samples[1, to_reflect]
            survivor_r2 = random_samples[2, to_reflect]
            
            for mat_id in np.unique(survivor_materials):
                material_mask = (survivor_materials == mat_id)
                survivor_dirs[:, material_mask] = self.reflections[mat_id].reflect(
                    survivor_dirs[:, material_mask], 
                    survivor_r1[material_mask], 
                    survivor_r2[material_mask]
                )
                
            batch.direction[:, to_reflect] = survivor_dirs
            
        return batch, to_reflect
    
    def is_below_ground(self, pos):
        return pos[Z] < 0
    
    def adjust_surface_boundary(self, batch: PhotonBatch):
        below_ground_mask = self.is_below_ground(batch.pos)
        batch.pos[:, below_ground_mask] += (0 - batch.pos[Z, below_ground_mask]) / batch.direction[Z, below_ground_mask] * batch.direction[:, below_ground_mask]
        batch.pos[Z, below_ground_mask] = -EPSILON
        return batch