import numpy as np
from atmorad.detectors.base import BaseDetector
from atmorad.environment.scene import Scene
from atmorad.engine.batch import PhotonBatch
from atmorad.config.config import SimConfig
from atmorad.constants import X, Y, Z, EPSILON

class BoundaryMapDetector(BaseDetector):
    def __init__(self):
        self.resolution = None
        self.toa_z = None
        self.domain_x = None
        self.domain_y = None
        self.surface_x, self.surface_y = [], []
        self.space_x, self.space_y = [], []
        self.x_edges = None
        self.y_edges = None

    def initialize(self, scene: Scene, config: SimConfig):
        self.toa_z = scene.atmosphere.top_of_atmosphere
        self.resolution = getattr(config.detectors, "map2d_resolution_km", 1.0)
        self.domain_x = config.geometry.domain_size_x_km
        self.domain_y = config.geometry.domain_size_y_km
        
        self.x_edges = np.arange(-self.domain_x/2, self.domain_x/2 + self.resolution, self.resolution)
        self.y_edges = np.arange(-self.domain_y/2, self.domain_y/2 + self.resolution, self.resolution)

    def record_termination(self, batch: PhotonBatch, terminated_mask: np.ndarray):
        if not np.any(terminated_mask):
            return

        term_pos = batch.pos[:, terminated_mask]
        
        # MAGIA GEOMETRII: Zawijanie (wraparound) fotonów na krawędziach domeny.
        # Foton, który ucieknie na X=60, zostanie "przeteleportowany" na X=-40.
        wrapped_x = np.mod(term_pos[X] + self.domain_x/2, self.domain_x) - self.domain_x/2
        wrapped_y = np.mod(term_pos[Y] + self.domain_y/2, self.domain_y) - self.domain_y/2

        # Używamy EPSILON*2 aby uodpornić się na drobne wahania float
        surface_mask = term_pos[Z] <= (EPSILON * 2)
        space_mask = term_pos[Z] >= (self.toa_z - EPSILON * 2)
        
        if np.any(surface_mask):
            self.surface_x.append(wrapped_x[surface_mask])
            self.surface_y.append(wrapped_y[surface_mask])
            
        if np.any(space_mask):
            self.space_x.append(wrapped_x[space_mask])
            self.space_y.append(wrapped_y[space_mask])

    def get_results(self) -> dict:
        results = {
            "x_edges": self.x_edges,
            "y_edges": self.y_edges
        }
        
        if self.surface_x:
            all_surf_x = np.concatenate(self.surface_x)
            all_surf_y = np.concatenate(self.surface_y)
            
            surf_map, _, _ = np.histogram2d(
                all_surf_x, all_surf_y, 
                bins=[self.x_edges, self.y_edges]
            )
            results["surface_flux_map_2d"] = surf_map
        else:
            results["surface_flux_map_2d"] = np.zeros((len(self.x_edges)-1, len(self.y_edges)-1))

        if self.space_x:
            all_space_x = np.concatenate(self.space_x)
            all_space_y = np.concatenate(self.space_y)
            space_map, _, _ = np.histogram2d(
                all_space_x, all_space_y, 
                bins=[self.x_edges, self.y_edges]
            )
            results["toa_flux_map_2d"] = space_map
        else:
            results["toa_flux_map_2d"] = np.zeros((len(self.x_edges)-1, len(self.y_edges)-1))

        return results