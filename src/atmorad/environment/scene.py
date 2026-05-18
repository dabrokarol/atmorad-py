import numpy as np

from atmorad.engine.batch import PhotonBatch
from atmorad.environment import Atmosphere, Surface
from atmorad.constants import EPSILON, X, Y, Z
from atmorad.physics import sun_zenith_to_direction
class Scene:
    def __init__(self, surface: Surface, atmosphere: Atmosphere) -> None:
        self.surface = surface
        self.atmosphere = atmosphere
    
    def process_interactions(self, batch: PhotonBatch, to_scatter_mask: np.ndarray, random_sample: np.ndarray) -> tuple[PhotonBatch, np.ndarray, np.ndarray, np.ndarray]:
        """
        Scatters and reflects photons.
        
        Args:
            batch: The current active PhotonBatch.
            random_samples: Array of shape (3, N) containing uniform random numbers 
                            for interaction type, theta, and phi respectively.
        """
        atmosphere_mask = self.in_atmosphere(batch.pos) & to_scatter_mask
        surface_mask = self.below_ground(batch.pos)
        to_scat = np.zeros_like(random_sample[0], dtype=bool)
        to_reflect = np.zeros_like(random_sample[0], dtype=bool)
        
        if np.any(atmosphere_mask):
            batch, to_scat = self.atmosphere.process_scattering(batch, atmosphere_mask, random_sample)
            
        if np.any(surface_mask):
            batch, to_reflect = self.surface.process_reflection(batch, surface_mask, random_sample)

        absorbed_surface = (~to_reflect) & surface_mask
        absorbed_atmosphere = (~to_scat) & atmosphere_mask
  
        return batch, absorbed_surface, absorbed_atmosphere, to_scat|to_reflect
        
    def tau_to_boundary(self, batch: PhotonBatch):
        return self.atmosphere.tau_to_boundary(batch)

    def above_toa(self, pos):
        return self.atmosphere.above_toa(pos)
    
    def below_ground(self, pos):
        return self.surface.is_below_ground(pos)
    
    def in_atmosphere(self, pos):
        return ~self.above_toa(pos) & ~self.below_ground(pos)
    
    def adjust_to_boundary_conditions(self, batch: PhotonBatch):
        batch = self.atmosphere.adjust_internal_boundaries(batch)
        batch = self.surface.adjust_surface_boundary(batch)
        return batch
    
    def start_pos(self, num_photons, rng):
        nx, ny = self.surface.domain_size
        pos = np.empty(shape=(3, num_photons), dtype=float)
        pos[X, :] = rng.uniform(-nx/2, nx/2, num_photons)
        pos[Y, :] = rng.uniform(-ny/2, ny/2, num_photons)
        pos[Z, :] = np.full(num_photons, self.atmosphere.top_of_atmosphere + EPSILON)
        return pos
    
    def start_direction(self, num_photons, theta_sun, phi_sun, rng):
        theta_sun_rad = theta_sun / 180 * np.pi
        phi_sun_rad = phi_sun / 180 * np.pi
        theta = rng.normal(theta_sun_rad, 1/60, size=num_photons)
        phi = rng.normal(phi_sun_rad, 1/60, size=num_photons)
        direction = sun_zenith_to_direction(theta, phi) 
        return direction
    
    def get_material_ids(self, pos, rng):
        rand_component = rng.uniform(0, 1, pos.shape[1])
        return self.atmosphere.get_material_ids(pos, rand_component)
    
    def move_photons(self, batch: PhotonBatch, tau_to_move: np.ndarray):
        dist = self.atmosphere.tau_to_distance(batch, tau_to_move)
        batch.pos += batch.direction * dist
        batch = self.adjust_to_boundary_conditions(batch)
        return batch
    
    def get_final_photon_position_data(self, pos):
        return self.above_toa(pos), self.below_ground(pos), self.atmosphere.get_spatial_indices(pos)