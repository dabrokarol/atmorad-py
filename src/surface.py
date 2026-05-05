import numpy as np

from src.physics.reflection import SurfaceReflection

class SurfaceMaterial:
    def __init__(self, albedo: float, reflection: SurfaceReflection):
        self.albedo = albedo
        self.reflection = reflection

class ProceduralMap:
    def __init__(self, procedure):
        self.procedure = procedure

    def get_material_ids(self, pos):
        return self.procedure(pos)
    
    #example map functions:
    @staticmethod
    def uniform_ground(pos):
        return np.zeros_like(pos[0]).astype(int)
    
    @staticmethod
    def split_half_x(pos):
        return np.where(pos[0] < 0, 0, 1)
    
    @staticmethod
    def checkerboard(pos):
        x = np.mod(pos[0], 10)
        y = np.mod(pos[1], 10)
        return np.where(((x<5) & (y<5)) | ((x >= 5) & (y >= 5)), 0, 1)

class GridMap:
    def __init__(self, ground_ids_matrix, periodic = True, grid_density=10):
        self.matrix = ground_ids_matrix
        self.periodic = periodic
        self.density = grid_density
        self.dims = ground_ids_matrix.shape / grid_density
        
    def get_material_ids(self, pos):
        if self.periodic:
            cords = np.mod(np.floor(pos[:2] * self.density).astype(int), self.dims[:, np.newaxis])
            return self.matrix[cords[0], cords[1]]
        else:
            raise FutureWarning('Non-periodic ground map is not yet implemented')

class Surface:
    def __init__(self, ground_map: GridMap | ProceduralMap, ground_types: list[SurfaceMaterial]):
        self.ground_map = ground_map
        self.ground_types = np.array(ground_types)
        self.albedos = np.array([material.albedo for material in self.ground_types])
        self.reflections = [material.reflection for material in self.ground_types]

    def check_reflection(self, pos, rand):
        ground_ids = self.ground_map.get_material_ids(pos)
        albedos = self.albedos[ground_ids]

        return albedos > rand
    
    def reflect(self, pos, ori, rand_1, rand_2):
        ground_ids = self.ground_map.get_material_ids(pos)
        result_ori = np.zeros_like(ori)

        for i, refl in enumerate(self.reflections):
            msk_i = ground_ids == i
            result_ori[:, msk_i] = refl.reflect(ori[:, msk_i], rand_1[msk_i], rand_2[msk_i])

        return result_ori
