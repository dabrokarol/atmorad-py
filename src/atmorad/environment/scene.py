import numpy as np

from atmorad.engine.batch import PhotonBatch
from atmorad.environment.atmosphere import LayeredAtmosphere
from atmorad.environment.surface import FlatSurface
class Scene:
    def __init__(self, surface: FlatSurface, atmosphere: LayeredAtmosphere) -> None:
        self.surface = surface
        self.atmosphere = atmosphere
        self.top_of_atmosphere = atmosphere.boundaries[-1]
    
    def process_interactions(self, batch: PhotonBatch, to_scatter_mask: np.ndarray, random_sample: np.ndarray) -> tuple[PhotonBatch, np.ndarray, np.ndarray, np.ndarray]:
        """
        Scatters and reflects photons.
        
        Args:
            batch: The current active PhotonBatch.
            random_samples: Array of shape (3, N) containing uniform random numbers 
                            for interaction type, theta, and phi respectively.
        """
        atmosphere_mask = self.in_atmosphere(batch.pos) & to_scatter_mask
        surface_mask = self.reached_surface(batch.pos)
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

    def reached_space(self, pos):
        return self.atmosphere.reached_space(pos)
    
    def reached_surface(self, pos):
        return self.surface.is_below_ground(pos)
    
    def in_atmosphere(self, pos):
        return ~self.reached_space(pos) & ~self.reached_surface(pos)
    
    def adjust_to_boundary_conditions(self, batch: PhotonBatch):
        batch = self.atmosphere.adjust_internal_boundaries(batch)
        batch = self.surface.adjust_surface_boundary(batch)
        return batch
    
    def move_photons(self, batch: PhotonBatch, tau_to_move: np.ndarray):
        dist = self.atmosphere.tau_to_distance(batch, tau_to_move)
        batch.pos += batch.direction * dist
        batch = self.adjust_to_boundary_conditions(batch)
        return batch
    
    def get_final_photon_position_data(self, pos):
        print(pos)
        print(self.top_of_atmosphere)
        print(self.reached_space(pos))
        print(self.reached_surface(pos))
        print(self.atmosphere._get_layer_idx(pos))
        return self.reached_space(pos), self.reached_surface(pos), self.atmosphere._get_layer_idx(pos)