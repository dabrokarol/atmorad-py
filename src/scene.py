import numpy as np
from .physics import orientation

from typing import Any, Iterable
from abc import ABC

class Scattering:
    def __init__(self, scatter_func, g, n_precomputed=1000):
        """Computes normalized probability distribution of cos_theta and sums to obtain distribuant."""
        cos_grid = np.linspace(-1, 1, n_precomputed)
        dx = cos_grid[1] - cos_grid[0]

        pdf = scatter_func(g, cos_grid)
        pdf /= (np.sum(pdf) * dx)

        self.distribuant = np.cumsum(pdf) * dx
        self.cos_grid = cos_grid
        self.n_precomputed = n_precomputed

    def scatter(self, rand_1, rand_2):
        """Computes sin and cos of theta, phi used for scattering. Uses `np.interp` to obtain reversed distribuant values for given rand_1. Samples phi from uniform distribution [0,2pi].
        
        Args:
            rand_1 - array of random numbers (uniform(0,1)) used to sample cos_theta
            rand_2 - array of random numbers (uniform(0,1)) used to sample sin_theta

        Returns:
            np.array((cos_t, sin_t, cos_p, sin_p)) - trigonometric functions of sampled angles
        """
        phi = 2*np.pi*rand_2
        cos_t = np.interp(rand_1, self.distribuant, self.cos_grid)
        sin_t = np.sqrt(1 - np.clip(cos_t**2, 0, 1))
        cos_p = np.cos(phi)
        sin_p = np.sin(phi)
        return np.array((cos_t, sin_t, cos_p, sin_p))
    
    def __call__(self, rand_1, rand_2) -> Any:
        return self.scatter(rand_1, rand_2)
    
    @staticmethod
    def henyey_greenstein(g, cos_t):
        """Henyey-Greenstein function."""
        if np.isclose(g, 1):
            return (np.isclose(cos_t, 1)).astype(float)
        elif np.isclose(g, -1):
            return (np.isclose(cos_t, -1)).astype(float)
        else:
            return (1 - g**2) / (2) / (1 + g**2 - 2*g*cos_t)**(3/2)
    @staticmethod    
    def uniform(g, cos_t):
        """Uniform scattering function."""
        return np.ones_like(cos_t, dtype=np.float64)
    
    @staticmethod
    def surf_henyey_greenstein(g, cos_t):
        """Modification of HG that scatters only for thetas > pi/2"""
        if np.isclose(g, -1):
            return (np.isclose(cos_t, -1)).astype(float)
        else:
            return np.where(
                cos_t < 0,
                (1 - g**2) / (2) / (1 + g**2 - 2*g*cos_t)**(3/2),
                0
            )
        
class SurfaceReflection:
    def __init__(self, reflection_func) -> None:
        self.reflection_func = reflection_func

    def reflect(self, ori, rand_1, rand_2):
        return self.reflection_func(ori, rand_1, rand_2)
    
    def __call__(self, ori, rand_1, rand_2):
        return self.reflection_func(ori, rand_1, rand_2)
    
    @staticmethod
    def mirror_reflection(ori, rand_1, rand_2):
        ori[2] = -ori[2]
        return ori
    
    @staticmethod
    def lambertian_reflection(ori, rand_1, rand_2):
        phi = rand_2 * 2 * np.pi

        cos_t = -np.sqrt(rand_1) # cosine-weighted hemisphere sampling, minus because surface has the biggest height
        sin_t = np.sqrt(1 - rand_1)
        cos_p = np.cos(phi)
        sin_p = np.sin(phi)

        return orientation(cos_t, sin_t, cos_p, sin_p)
    
    @staticmethod
    def uniform_reflection(ori, rand_1, rand_2):
        theta = rand_1 * np.pi / 2
        phi = rand_2 * 2 * np.pi

        cos_t = -np.cos(theta) # uniform sampling
        sin_t = np.sin(theta)
        cos_p = np.cos(phi)
        sin_p = np.sin(phi)

        return orientation(cos_t, sin_t, cos_p, sin_p)
        
class AtmosphericMedium:
    def __init__(self, mu: float, albedo: float, scattering: Scattering):
        self.albedo = albedo
        self.mu = mu
        self.scattering = scattering

class AtmosphericLayer:
    def __init__(self, height, ingredients: list[tuple[AtmosphericMedium, float]]):
        self.height = height

        p = 0
        for i in ingredients:
            p += i[1]
        if not np.isclose(p, 1):
            raise ValueError("Initialized layer probabilities don't sum to one. Check atmospheric layers initialization.")

        self.ingredients = ingredients

class Atmosphere:
    def __init__(self, layers: list[AtmosphericLayer]):
        boundaries = [0]
        unique_mediums = []
        max_layer_width = 0
        for l in layers:
            boundaries.append(l.height)
            for medium, probability in l.ingredients:
                if not medium in unique_mediums:
                    unique_mediums.append(medium)
            max_layer_width = max(max_layer_width, len(l.ingredients))

        layer_cdfs = np.zeros((len(layers), max_layer_width))
        layer_medium_ids = np.zeros((len(layers), max_layer_width)).astype(int)

        for i, l in enumerate(layers):
            for j, (medium, probability) in enumerate(l.ingredients):
                if j == 0:
                    layer_cdfs[i, j] = probability
                else:
                    layer_cdfs[i, j] = layer_cdfs[i, j-1] + probability # cumulative summation to obtain cumulative density function
                layer_medium_ids[i, j] = unique_mediums.index(medium)
            width = len(l.ingredients)
            items_left = max_layer_width - width
            for j in range(width, width + items_left):
                layer_cdfs[i, j] = layer_cdfs[i, j-1]
                layer_medium_ids[i, j] = layer_medium_ids[i, j-1]

        self.albedos = np.array([medium.albedo for medium in unique_mediums])
        self.mus = np.array([medium.mu for medium in unique_mediums])
        self.boundaries = np.cumsum(boundaries)

        self.scatterings = [medium.scattering for medium in unique_mediums]

        self.layer_cdfs = layer_cdfs
        self.layer_medium_ids = layer_medium_ids

    def get_layer_idx(self, pos_z):
        layer_idx = np.searchsorted(self.boundaries, pos_z, 'left')
        layer_idx[pos_z < 0] = -1 # space will give -1
        return layer_idx

    def get_mediums(self, pos, rand_1):
        layer_idx = np.searchsorted(self.boundaries, pos[2])
        layer_medium_idx = np.argmax(rand_1[:, np.newaxis] < self.layer_cdfs[layer_idx], axis=1)
        return self.layer_medium_ids[layer_idx, layer_medium_idx] # array of column numbers, array of row numbers

    def check_scat(self, medium_ids, rand_1):
        return rand_1 < self.albedos[medium_ids]
    
    def scatter(self, medium_ids, rand_t, rand_p):
        scat_trig = np.zeros((4, rand_t.size))
        for i, scat in enumerate(self.scatterings):
            msk_i = medium_ids == i
            scat_trig[:, msk_i] = scat(rand_t[msk_i], rand_p[msk_i])
        return scat_trig

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
        return np.zeros_like(pos.shape[1])
    
    @staticmethod
    def split_half_x(pos):
        return np.where(pos[0] < 0, 0, 1)
    
    @staticmethod
    def checkerboard(pos):
        x = np.mod(pos[0], 10)
        y = np.mod(pos[1], 10)
        return np.where((x<5 & y<5) | (x >= 5 & y >= 5), 0, 1)

class GridMap:
    def __init__(self, ground_ids_matrix, periodic = True, grid_density=10, dimensions=(10,10)):
        self.matrix = ground_ids_matrix
        self.periodic = periodic
        self.density = grid_density
        self.dims = np.array(dimensions)
        
    def get_material_ids(self, pos):
        if self.periodic:
            cords = np.mod(np.floor(pos[:2] * self.density).astype(int), self.dims[:, np.newaxis])
            return self.matrix[cords[0], cords[1]]
        else:
            raise FutureWarning('Non-periodic ground map is not yet implemented')

class Surface:
    def __init__(self, ground_map: GridMap | ProceduralMap, ground_types = Iterable[SurfaceMaterial]):
        self.ground_map = ground_map
        self.ground_types = np.array(ground_types)

        self.albedos = [material.albedo for material in self.ground_types]
        self.reflections = [material.reflection for material in self.ground_types]

    def check_reflection(self, pos, rand):
        ground_ids = self.ground_map.get_material_ids(pos)
        albedos = self.albedos[ground_ids]

        return albedos < rand
    
    def reflect(self, pos, ori, rand_1, rand_2):
        ground_ids = self.ground_map.get_material_ids(pos)
        result_ori = np.zeros_like(ori)

        for i, refl in enumerate(self.reflections):
            msk_i = ground_ids == i
            result_ori[:, msk_i] = refl.reflect(ori[:, msk_i], rand_1[msk_i], rand_2[msk_i])

        return result_ori

class Space:
    def __init__(self):
        pass


class Scene:
    def __init__(self, surface: Surface, atmosphere: Atmosphere, space: Space, config: dict) -> None:
        self.surface = surface
        self.space = space
        self.atmosphere = atmosphere

    def move_photons(self, pos, ori, tau_to_travel):
        """Function moves photons according to their tau_to_travel.
        
        Args:
            pos: array of shape (3, N) - positions of photons
            ori: array of shape (3, N) - unit-vectors in direction of orientation 
            tau_to_travel: array of shape (N) - optical depth to travel

        Returns:
            final_pos, surface_msk, space_msk
            """
        boundaries = self.boundaries
        mus = np.array([l.mu for l in self.layers])

        ids = np.arange(0, pos.shape[1])

        final_pos = np.zeros_like(pos)
        final_space_msk = np.zeros_like(ids, dtype=bool)
        final_surface_msk = np.zeros_like(ids, dtype=bool)

        while tau_to_travel.size:
            layer_idx = self.get_layer_idx(pos[2])

            surface_msk = layer_idx == self.boundaries.size
            space_msk = layer_idx == -1
            atmoshpere_msk = (~space_msk) & (~surface_msk)

            pos = self.snap_to_boundaries(pos, ori, space_msk, surface_msk)

            final_pos[:, ids[~atmoshpere_msk]] = pos[:, ~atmoshpere_msk]
            final_space_msk[ids[space_msk]] = True
            final_surface_msk[ids[surface_msk]] = True

            # shrink masks
            layer_idx = layer_idx[atmoshpere_msk]
            tau_to_travel = tau_to_travel[atmoshpere_msk]
            ori = ori[:, atmoshpere_msk]
            pos = pos[:, atmoshpere_msk]
            ids = ids[atmoshpere_msk]

            relative_pos_z = pos[2] - self.boundaries[layer_idx]
            travel_to_space = ori[2] < 0
            travel_to_ground = ori[2] > 0
            travel_horizontal = ori[2] == 0
            ori[2, travel_horizontal] = 1e-12 # to avoid dividing by zero

            lower_bound = boundaries[layer_idx]
            upper_bound = boundaries[layer_idx + 1]

            relative_pos_z = np.zeros_like(pos[2])
            relative_pos_z[travel_to_space] = lower_bound[travel_to_space] - pos[2, travel_to_space]
            relative_pos_z[travel_to_ground] = upper_bound[travel_to_ground] - pos[2, travel_to_ground]
            relative_pos_z[travel_horizontal] = np.inf


            mu = mus[layer_idx]
            tau_to_boundary = (relative_pos_z) / ori[2] * mu

            new_tau_to_travel = np.where(tau_to_boundary < tau_to_travel, tau_to_boundary, tau_to_travel)
            dist = new_tau_to_travel / mu
            pos += ori * dist

            tau_to_travel -= new_tau_to_travel
            finished_mask = np.isclose(tau_to_travel, 0)

            final_pos[:, ids[finished_mask]] = pos[:, finished_mask]

            layer_idx = layer_idx[~finished_mask]
            tau_to_travel = tau_to_travel[~finished_mask]
            ori = ori[:, ~finished_mask]
            pos = pos[:, ~finished_mask]
            ids = ids[~finished_mask]

        return final_pos, final_surface_msk, final_space_msk
    
    def scatter_photons(self, pos, ori, rand_1, rand_2, rand_3, surface_mask, atmosphere_mask, in_cloud_mask):
        """Scatters and reflects photons based on random numbers."""
        in_air_mask = (~in_cloud_mask) & atmosphere_mask
        air_layer_idx = np.where(in_air_mask, self.get_layer_idx(pos[2]), 0)
        cloud_layer_idx = np.where(in_cloud_mask, self.get_layer_idx(pos[2]), 0)

        albedos = np.zeros_like(rand_1.size)
        albedos[in_air_mask] = self.air_albedos[air_layer_idx]
        albedos[in_cloud_mask] = self.cloud_albedos[cloud_layer_idx]
        albedos[surface_mask] = self.surface_albedos

        for i, layer in enumerate(self.layers):
            air_scattered = rand_1[air_layer_idx == i] < layer.ss_albedo
            cloud_scattered = rand_1[cloud_layer_idx == i] < layer.cloud_ss_albedo

    
    def snap_to_boundaries(self, pos, ori, reached_space, reached_surf):
        pos[:, reached_space] += (0 - pos[2, reached_space]) / ori[2, reached_space] * ori[:, reached_space]
        pos[:, reached_surf] += (self.boundaries[-1] - pos[2, reached_surf]) / ori[2, reached_surf] * ori[:, reached_surf]
        return pos