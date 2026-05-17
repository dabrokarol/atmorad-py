import numpy as np
from atmorad.detectors.base import BaseDetector
from atmorad.environment.scene import Scene
from atmorad.engine.batch import PhotonBatch
from atmorad.config.config import SimConfig
from atmorad.constants import X, Y, Z

class IncidentFluxMapDetector(BaseDetector):
    def __init__(self):
        self.target_z_array = None
        
        self.domain_x = None
        self.domain_y = None
        self.resolution = None
        
        self.hit_p_down, self.hit_x_down, self.hit_y_down = [], [], []
        self.hit_p_up, self.hit_x_up, self.hit_y_up = [], [], []
        
        self.x_edges = None
        self.y_edges = None

    def initialize(self, scene: Scene, config: SimConfig):
        
        self.target_z_array = np.array(config.detectors.incident_flux_heights_km, dtype=float)
        self.resolution = getattr(config.detectors, "map2d_resolution_km", 1.0)
        self.domain_x = config.geometry.domain_size_x_km
        self.domain_y = config.geometry.domain_size_y_km
        
        self.x_edges = np.arange(-self.domain_x/2, self.domain_x/2 + self.resolution, self.resolution)
        self.y_edges = np.arange(-self.domain_y/2, self.domain_y/2 + self.resolution, self.resolution)

    def _process_hits(self, batch: PhotonBatch, old_pos: np.ndarray, crossed_mask: np.ndarray, 
                      p_list: list, x_list: list, y_list: list):
        if not np.any(crossed_mask):
            return

        photon_idx, plane_idx = np.where(crossed_mask)
        
        crossed_old_z = old_pos[Z, photon_idx]
        crossed_dir_z = batch.direction[Z, photon_idx]
        crossed_target_z = self.target_z_array[plane_idx]

        with np.errstate(divide='ignore', invalid='ignore'):
            t = (crossed_target_z - crossed_old_z) / crossed_dir_z
        
        exact_x = old_pos[X, photon_idx] + batch.direction[X, photon_idx] * t
        exact_y = old_pos[Y, photon_idx] + batch.direction[Y, photon_idx] * t
        
        wrapped_x = np.mod(exact_x + self.domain_x/2, self.domain_x) - self.domain_x/2
        wrapped_y = np.mod(exact_y + self.domain_y/2, self.domain_y) - self.domain_y/2

        p_list.append(plane_idx)
        x_list.append(wrapped_x)
        y_list.append(wrapped_y)

    def record_movement(self, batch: PhotonBatch, old_pos: np.ndarray):
        old_z = old_pos[Z]
        new_z = batch.pos[Z]

        down_mask = (old_z[:, np.newaxis] > self.target_z_array) & (new_z[:, np.newaxis] <= self.target_z_array)
        up_mask = (old_z[:, np.newaxis] < self.target_z_array) & (new_z[:, np.newaxis] >= self.target_z_array)

        self._process_hits(batch, old_pos, down_mask, self.hit_p_down, self.hit_x_down, self.hit_y_down)
        self._process_hits(batch, old_pos, up_mask, self.hit_p_up, self.hit_x_up, self.hit_y_up)

    def _build_maps(self, hit_p: list, hit_x: list, hit_y: list) -> dict:
        flux_maps = {}
        if hit_x:
            all_p = np.concatenate(hit_p)
            all_x = np.concatenate(hit_x)
            all_y = np.concatenate(hit_y)
            
            for i, z_val in enumerate(self.target_z_array):
                p_mask = (all_p == i)
                flux_map, _, _ = np.histogram2d(
                    all_x[p_mask], all_y[p_mask], bins=[self.x_edges, self.y_edges]
                )
                flux_maps[float(z_val)] = flux_map
        else:
            for z_val in self.target_z_array:
                flux_maps[float(z_val)] = np.zeros((len(self.x_edges)-1, len(self.y_edges)-1))
                
        return flux_maps

    def get_results(self) -> dict:
        return {
            "x_edges": self.x_edges,
            "y_edges": self.y_edges,
            "incident_flux_down_maps_2d": self._build_maps(self.hit_p_down, self.hit_x_down, self.hit_y_down),
            "incident_flux_up_maps_2d": self._build_maps(self.hit_p_up, self.hit_x_up, self.hit_y_up),
            "incident_flux_heights_km": list(self.target_z_array)
        }