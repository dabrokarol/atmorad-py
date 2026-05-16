import numpy as np

from typing import Sequence
from abc import ABC, abstractmethod
from dataclasses import dataclass

from atmorad.physics.reflection import SurfaceReflection
from atmorad.constants import X, Y, Z

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
    def circle(pos):
        return np.where((pos[X]**2 + pos[Y]**2) < 100, 0, 1)
    
    @staticmethod
    def checkerboard(pos):
        x = np.mod(pos[X], 10)
        y = np.mod(pos[Y], 10)
        return np.where(((x<5) & (y<5)) | ((x >= 5) & (y >= 5)), 0, 1)

# class GridMap(SurfaceMap):
#     def __init__(self, ground_ids_matrix, periodic = True, grid_density=10):
#         self.matrix = ground_ids_matrix
#         self.periodic = periodic
#         self.density = grid_density
#         self.dims = np.array(ground_ids_matrix.shape) / grid_density
        
#     def get_material_ids(self, pos):
#         if self.periodic:
#             cords = np.mod(np.floor(pos[:2] * self.density).astype(int), self.dims[:, np.newaxis])
#             return self.matrix[cords[0], cords[1]]
#         else:
#             raise NotImplementedError('Non-periodic ground map is not yet implemented')

class Surface:
    def __init__(self, ground_map: SurfaceMap, ground_types: Sequence[SurfaceMaterial]):
        self.ground_map = ground_map
        self.ground_types = np.array(ground_types)
        self.albedos = np.array([material.albedo for material in self.ground_types])
        self.reflections = [material.reflection for material in self.ground_types]
    
    def process_reflection(self, pos, direction, rand_albedo, rand_theta, rand_phi):
        ground_ids = self.ground_map.get_material_ids(pos)
        
        albedos = self.albedos[ground_ids]
        reflect_mask = albedos > rand_albedo

        result_direction = np.zeros_like(direction)

        for i, refl in enumerate(self.reflections):
            mask_i = (ground_ids == i) & reflect_mask

            result_direction[:, mask_i] = refl.reflect(direction[:, mask_i], rand_theta[mask_i], rand_phi[mask_i])

        return reflect_mask, result_direction
