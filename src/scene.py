import numpy as np

from src.physics import rotate
from src.atmosphere import Atmosphere
from src.surface import Surface

class Space:
    def __init__(self):
        pass

class Scene:
    def __init__(self, surface: Surface, atmosphere: Atmosphere, space: Space, config: dict) -> None:
        self.surface = surface
        self.space = space
        self.atmosphere = atmosphere

    def move_photons(self, pos, ori, tau_to_travel, rng):
        """Function moves photons according to their tau_to_travel.
        
        Args:
            pos: array of shape (3, N) - positions of photons
            ori: array of shape (3, N) - unit-vectors in direction of orientation 
            tau_to_travel: array of shape (N) - optical depth to travel

        Returns:
            final_pos, surface_msk, space_msk
            """
        boundaries = self.atmosphere.boundaries
        ids = np.arange(0, pos.shape[1])

        final_pos = np.zeros_like(pos)
        final_space_msk = np.zeros_like(ids, dtype=bool)
        final_surface_msk = np.zeros_like(ids, dtype=bool)

        rand_weather = rng.uniform(0, 1, ids.size)
        medium_ids = self.atmosphere.get_mediums(pos, rand_weather)

        while tau_to_travel.size:
            layer_idx = self.atmosphere.get_layer_idx(pos[2])

            surface_msk = pos[2] > boundaries[-1]
            space_msk = pos[2] < 0 
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

            relative_pos_z = pos[2] - self.atmosphere.boundaries[layer_idx]
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

            mu = self.atmosphere.mus[medium_ids[ids]]
            tau_to_boundary = (relative_pos_z) / ori[2] * mu

            new_tau_to_travel = np.where(tau_to_boundary < tau_to_travel, tau_to_boundary, tau_to_travel)
            dist = new_tau_to_travel / mu
            dist[tau_to_boundary < tau_to_travel] += 1e-8
            pos += ori * dist

            tau_to_travel -= new_tau_to_travel
            finished_mask = np.isclose(tau_to_travel, 0)

            in_atmosphere_mask = (pos[2] >= 0) & (pos[2] <   self.atmosphere.boundaries[-1])
            cross_layer_mask = ~finished_mask & in_atmosphere_mask
            n_cross_layer = np.count_nonzero(cross_layer_mask)
            rand_weather = rng.uniform(0, 1, n_cross_layer)
            medium_ids[ids[cross_layer_mask]] = self.atmosphere.get_mediums(pos[:, cross_layer_mask], rand_weather)

            final_pos[:, ids[finished_mask]] = pos[:, finished_mask]

            layer_idx = layer_idx[~finished_mask]
            tau_to_travel = tau_to_travel[~finished_mask]
            ori = ori[:, ~finished_mask]
            pos = pos[:, ~finished_mask]
            ids = ids[~finished_mask]

        return final_pos, final_surface_msk, final_space_msk, medium_ids
    
    def scatter_photons(self, pos, ori, rand_1, rand_2, rand_3, surface_mask, atmosphere_mask, medium_ids):
        """Scatters and reflects photons based on random numbers."""
        to_scat = np.zeros_like(rand_1).astype(bool)
        to_scat[atmosphere_mask] = self.atmosphere.check_scat(medium_ids[atmosphere_mask], rand_1[atmosphere_mask])
        cos_t, sin_t, cos_p, sin_p = self.atmosphere.scatter(medium_ids[to_scat], rand_2[to_scat], rand_3[to_scat])

        ori[:, to_scat] = rotate(ori[:, to_scat], cos_t, sin_t, cos_p, sin_p)

        to_reflect = np.zeros_like(rand_1).astype(bool)
        to_reflect[surface_mask] = self.surface.check_reflection(pos[:, surface_mask], rand_1[surface_mask])
        ori[:, to_reflect] = self.surface.reflect(pos[:, to_reflect], ori[:, to_reflect], rand_2[to_reflect], rand_3[to_reflect])

        absorbed_surface = (~to_reflect) & surface_mask
        absorbed_atmosphere = (~to_scat) & atmosphere_mask

        return ori, absorbed_surface, absorbed_atmosphere
    
    def snap_to_boundaries(self, pos, ori, reached_space, reached_surf):
        pos[:, reached_space] += (0 - pos[2, reached_space]) / ori[2, reached_space] * ori[:, reached_space]
        pos[:, reached_surf] += (self.atmosphere.boundaries[-1] - pos[2, reached_surf]) / ori[2, reached_surf] * ori[:, reached_surf]
        return pos