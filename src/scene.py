import numpy as np

from src.physics import rotate
from src.atmosphere import Atmosphere
from src.surface import Surface

class Space:
    def __init__(self):
        pass

class Scene:
    def __init__(self, surface: Surface, atmosphere: Atmosphere, space: Space) -> None:
        self.surface = surface
        self.space = space
        self.atmosphere = atmosphere

    def move_photons(self, pos, direction, tau_to_travel, rng):
        """Function moves photons according to their tau_to_travel.
        
        Args:
            pos: array of shape (3, N) - positions of photons
            direction: array of shape (3, N) - unit-vectors in direction of orientation 
            tau_to_travel: array of shape (N) - optical depth to travel

        Returns:
            final_pos, surface_mask, space_mask
            """
        boundaries = self.atmosphere.boundaries
        ids = np.arange(0, pos.shape[1])

        final_pos = np.zeros_like(pos)
        final_space_mask= np.zeros_like(ids, dtype=bool)
        final_surface_mask= np.zeros_like(ids, dtype=bool)

        rand_component = rng.uniform(0, 1, ids.size)
        medium_ids = self.atmosphere.get_mediums(pos, rand_component)

        while tau_to_travel.size:
            layer_idx = self.atmosphere.get_layer_idx(pos[2])

            surface_mask= pos[2] > boundaries[-1]
            space_mask= pos[2] < 0 
            atmosphere_mask = (~space_mask) & (~surface_mask)

            pos = self.snap_to_boundaries(pos, direction, space_mask, surface_mask)

            final_pos[:, ids[~atmosphere_mask]] = pos[:, ~atmosphere_mask]
            final_space_mask[ids[space_mask]] = True
            final_surface_mask[ids[surface_mask]] = True

            # shrink masks
            layer_idx = layer_idx[atmosphere_mask]
            tau_to_travel = tau_to_travel[atmosphere_mask]
            direction = direction[:, atmosphere_mask]
            pos = pos[:, atmosphere_mask]
            ids = ids[atmosphere_mask]

            delta_z = pos[2] - self.atmosphere.boundaries[layer_idx]
            travel_up = direction[2] < 0
            travel_down = direction[2] > 0
            travel_horizontal = direction[2] == 0
            direction[2, travel_horizontal] = 1e-12 # to avoid dividing by zero

            lower_bound = boundaries[layer_idx]
            upper_bound = boundaries[layer_idx + 1]

            delta_z[travel_up] = lower_bound[travel_up] - pos[2, travel_up]
            delta_z[travel_down] = upper_bound[travel_down] - pos[2, travel_down]
            delta_z[travel_horizontal] = np.inf

            excinction_coeff = self.atmosphere.extinction_coeffs[medium_ids[ids]]
            tau_to_boundary = (delta_z) / direction[2] * excinction_coeff

            new_tau_to_travel = np.where(tau_to_boundary < tau_to_travel, tau_to_boundary, tau_to_travel)
            dist = new_tau_to_travel / excinction_coeff
            dist[tau_to_boundary < tau_to_travel] += 1e-8
            pos += direction * dist

            tau_to_travel -= new_tau_to_travel
            finished_mask = np.isclose(tau_to_travel, 0)

            in_atmosphere_mask = (pos[2] >= 0) & (pos[2] <   self.atmosphere.boundaries[-1])
            cross_layer_mask = ~finished_mask & in_atmosphere_mask
            n_cross_layer = np.count_nonzero(cross_layer_mask)
            rand_component = rng.uniform(0, 1, n_cross_layer)
            medium_ids[ids[cross_layer_mask]] = self.atmosphere.get_mediums(pos[:, cross_layer_mask], rand_component)

            final_pos[:, ids[finished_mask]] = pos[:, finished_mask]

            layer_idx = layer_idx[~finished_mask]
            tau_to_travel = tau_to_travel[~finished_mask]
            direction = direction[:, ~finished_mask]
            pos = pos[:, ~finished_mask]
            ids = ids[~finished_mask]

        return final_pos, final_surface_mask, final_space_mask, medium_ids
    
    def scatter_photons(self, pos, direction, rand_interaction, rand_theta, rand_phi, surface_mask, atmosphere_mask, medium_ids):
        """Scatters and reflects photons based on random numbers."""
        to_scat = np.zeros_like(rand_interaction).astype(bool)
        to_scat[atmosphere_mask] = self.atmosphere.is_scattered(medium_ids[atmosphere_mask], rand_interaction[atmosphere_mask])
        cos_theta, sin_theta, cos_phi, sin_phi = self.atmosphere.scatter(medium_ids[to_scat], rand_theta[to_scat], rand_phi[to_scat])

        direction[:, to_scat] = rotate(direction[:, to_scat], cos_theta, sin_theta, cos_phi, sin_phi)

        to_reflect = np.zeros_like(rand_interaction).astype(bool)
        to_reflect[surface_mask] = self.surface.check_reflection(pos[:, surface_mask], rand_interaction[surface_mask])
        direction[:, to_reflect] = self.surface.reflect(pos[:, to_reflect], direction[:, to_reflect], rand_theta[to_reflect], rand_phi[to_reflect])

        absorbed_surface = (~to_reflect) & surface_mask
        absorbed_atmosphere = (~to_scat) & atmosphere_mask

        return direction, absorbed_surface, absorbed_atmosphere, to_scat
    
    def snap_to_boundaries(self, pos, direction, reached_space, reached_surf):
        pos[:, reached_space] += (0 - pos[2, reached_space]) / direction[2, reached_space] * direction[:, reached_space]
        pos[:, reached_surf] += (self.atmosphere.boundaries[-1] - pos[2, reached_surf]) / direction[2, reached_surf] * direction[:, reached_surf]
        return pos
    
    def get_photon_position_mask(self, pos_z):
        space_mask= pos_z <= 0
        surface_mask= pos_z >= self.atmosphere.boundaries[-1]
        layer_idx = self.atmosphere.get_layer_idx(pos_z)
        return space_mask, surface_mask, layer_idx