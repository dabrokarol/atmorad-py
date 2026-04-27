import numpy as np
from .physics import orientation

from typing import Iterable
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
            cos_t, sin_t, cos_p, sin_p - trigonometric functions of sampled angles
        """
        phi = 2*np.pi*rand_2
        cos_t = np.interp(rand_1, self.distribuant, self.cos_grid)
        sin_t = np.sqrt(1 - np.clip(cos_t**2, 0, 1))
        cos_p = np.cos(phi)
        sin_p = np.sin(phi)
        return cos_t, sin_t, cos_p, sin_p
    
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
        


class Atmosphere:
    def __init__(self, tau, height, albedo, scattering: Scattering):

        self.scattering = scattering
        self.ss_albedo = albedo
        self.tau = tau
        self.height = height
        self.mu = self.tau / self.height

    def check_scat(self, pos, ori, rand):
        return rand < self.ss_albedo # uniform albedo for now
    
    def scatter(self, pos, ori, rand_t, rand_p):
        # future: it will be able to use different scatterign functions based on position
        return self.scattering.scatter(rand_t, rand_p)


class Surface:
    def __init__(self, ground_ids: np.ndarray, albedos: Iterable[float], reflections: Iterable[SurfaceReflection]):
        self.ground_id = ground_ids
        self.albedos = albedos
        self.reflections = reflections

        self.dimensions = np.array(ground_ids.shape)
        self.grid_density = 10
        self.periodic = True

    def _snap_grid(self, pos):

        if self.periodic:
            snapped = np.floor(pos[:2] * self.grid_density).astype(int)
            return np.mod(snapped, self.dimensions[:, np.newaxis])
        else:
            raise FutureWarning('behaviour for non-periodic conditions is not yet implemented')
            # return np.floor(pos * self.grid_density)

    def check_reflection(self, pos, rand):
        
        cords = self._snap_grid(pos)

        if cords.size == 0:
            return np.zeros_like(rand, dtype=bool)
        
        ground_type = self.ground_id[cords[0], cords[1]]
        result_refl = np.zeros_like(rand)

        for i, albedo in enumerate(self.albedos):
            msk_i = (ground_type == i)
            result_refl[msk_i] = rand[msk_i] < albedo

        return result_refl
    
    def reflect(self, pos, ori, rand_1, rand_2):

        cords = self._snap_grid(pos)

        if cords.size == 0:
            return ori
        
        ground_type = self.ground_id[cords[0], cords[1]]
        result_ori = np.zeros_like(ori)

        for i, refl in enumerate(self.reflections):
            msk_i = ground_type == i
            result_ori[:, msk_i] = refl.reflect(ori[:, msk_i], rand_1[msk_i], rand_2[msk_i])

        return result_ori

class Space:
    def __init__(self):
        pass


class Scene:
    def __init__(self, surface: Surface, space: Space, layers: Iterable[Atmosphere], config: dict) -> None:
        self.surface = surface
        self.space = space
        self.layers = layers

        boundaries = [0]
        for l in layers:
            boundaries.append(l.height)
        self.boundaries = np.cumsum(boundaries)

        self.left_space = 0
        self.absorbed_ground = 0
        self.absorbed_atmosphere = 0

    def get_layer_idx(self, pos_z):
        layer_idx = np.searchsorted(self.boundaries, pos_z, 'left')
        layer_idx[pos_z < 0] = -1 # space will give is -1
        return layer_idx

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
    
    def scatter_photons(self, pos, ori, rand_1, rand_2, rand_3, surface_mask, atmosphere_mask):
        """Scatters and reflects photons based on random numbers."""
        pass
    
    def snap_to_boundaries(self, pos, ori, reached_space, reached_surf):
        pos[:, reached_space] += (0 - pos[2, reached_space]) / ori[2, reached_space] * ori[:, reached_space]
        pos[:, reached_surf] += (self.boundaries[-1] - pos[2, reached_surf]) / ori[2, reached_surf] * ori[:, reached_surf]
        return pos

        
    def check_photons(self, pos, orientation):
        """Checks whether photons left the map and snaps their possitions accordingly."""
        pass

    def photons_travel(self, pos, orientation, tau):
        """Moves photons to adequate layers based on tau that they should travel."""
        pass
