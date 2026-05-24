import numpy as np

from atmorad.constants import X, Y, Z
from atmorad.models import PhotonBatch
from atmorad.physics import sun_zenith_to_direction

from .atmosphere import Atmosphere
from .surface import BaseSurface


class Scene:
    def __init__(self, surface: BaseSurface, atmosphere: Atmosphere) -> None:
        self.surface = surface
        self.atmosphere = atmosphere

    def process_interactions(
        self,
        batch: PhotonBatch,
        scatter_mask: np.ndarray,
        surface_mask: np.ndarray,
        rng: np.random.Generator,
    ) -> PhotonBatch:
        """
        Scatters and reflects photons.

        Args:
            batch: The current active PhotonBatch.
            random_samples: Array of shape (3, N) containing uniform random numbers
                            for interaction type, theta, and phi respectively.
        """
        atmosphere_mask = self.in_atmosphere(batch.pos) & scatter_mask

        if np.any(atmosphere_mask):
            batch = self.atmosphere.process_scattering(batch, atmosphere_mask, rng)

        if np.any(surface_mask):
            batch = self.surface.process_reflection(batch, surface_mask, rng)

        return batch

    def tau_to_boundary(self, batch: PhotonBatch):
        return self.atmosphere.tau_to_boundary(batch)

    def above_toa(self, pos):
        return self.atmosphere.above_toa(pos)

    def at_surface(self, pos):
        return self.surface.crossed_ground(pos)

    def in_atmosphere(self, pos):
        return ~self.above_toa(pos) & ~self.at_surface(pos)

    def adjust_to_boundary_conditions(self, batch: PhotonBatch):
        batch = self.atmosphere.adjust_internal_boundaries(batch)
        batch = self.surface.adjust_surface_boundary(batch)
        return batch

    def start_pos(self, num_photons, rng: np.random.Generator):
        nx, ny = self.surface.domain_size
        pos = np.empty(shape=(3, num_photons), dtype=float)
        pos[X, :] = rng.uniform(-nx / 2, nx / 2, num_photons)
        pos[Y, :] = rng.uniform(-ny / 2, ny / 2, num_photons)
        pos[Z, :] = np.full(num_photons, self.atmosphere.top_of_atmosphere)
        return pos

    def start_direction(self, num_photons, theta_sun, phi_sun):
        theta = theta_sun / 180 * np.pi
        phi = phi_sun / 180 * np.pi
        direction = np.repeat(
            sun_zenith_to_direction(theta, phi)[:, np.newaxis], num_photons, axis=1
        )
        return direction

    def get_material_ids(self, pos, rng: np.random.Generator):
        rand_component = rng.uniform(0, 1, pos.shape[1])
        return self.atmosphere.get_material_ids(pos, rand_component)

    def move_photons(self, batch: PhotonBatch, tau_to_move: np.ndarray):
        dist = self.atmosphere.tau_to_distance(batch, tau_to_move)
        batch.pos += batch.direction * dist
        batch = self.adjust_to_boundary_conditions(batch)
        return batch

    def get_final_photon_position_data(self, pos):
        return self.above_toa(pos), self.at_surface(pos), self.atmosphere.get_spatial_indices(pos)
