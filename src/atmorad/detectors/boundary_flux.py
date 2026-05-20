import numpy as np

from atmorad.config import SimConfig
from atmorad.constants import DETECTOR_OFFSET, X, Y, Z
from atmorad.environment import Scene
from atmorad.models import PhotonBatch

from .base import BaseDetector


class BoundaryAbsorptionDetector(BaseDetector):
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

        num_bins_x = int(np.round(self.domain_x / self.resolution))
        num_bins_y = int(np.round(self.domain_y / self.resolution))

        self.x_edges = np.linspace(-self.domain_x / 2, self.domain_x / 2, num_bins_x + 1)
        self.y_edges = np.linspace(-self.domain_y / 2, self.domain_y / 2, num_bins_y + 1)

    def record_termination(self, batch: PhotonBatch, terminated_mask: np.ndarray):
        if not np.any(terminated_mask):
            return

        term_pos = batch.pos[:, terminated_mask]

        wrapped_x = np.mod(term_pos[X] + self.domain_x / 2, self.domain_x) - self.domain_x / 2
        wrapped_y = np.mod(term_pos[Y] + self.domain_y / 2, self.domain_y) - self.domain_y / 2

        surface_mask = term_pos[Z] <= DETECTOR_OFFSET
        above_toa_mask = term_pos[Z] >= (self.toa_z - DETECTOR_OFFSET)

        if np.any(surface_mask):
            self.surface_x.append(wrapped_x[surface_mask])
            self.surface_y.append(wrapped_y[surface_mask])

        if np.any(above_toa_mask):
            self.space_x.append(wrapped_x[above_toa_mask])
            self.space_y.append(wrapped_y[above_toa_mask])

    def get_results(self) -> dict:
        results = {"x_edges": self.x_edges, "y_edges": self.y_edges}

        if self.surface_x:
            all_surf_x = np.concatenate(self.surface_x)
            all_surf_y = np.concatenate(self.surface_y)

            surf_map, _, _ = np.histogram2d(
                all_surf_x, all_surf_y, bins=[self.x_edges, self.y_edges]
            )
            results["surface_absorption_map_2d"] = surf_map
        else:
            results["surface_absorption_map_2d"] = np.zeros(
                (len(self.x_edges) - 1, len(self.y_edges) - 1)
            )

        if self.space_x:
            all_space_x = np.concatenate(self.space_x)
            all_space_y = np.concatenate(self.space_y)
            space_map, _, _ = np.histogram2d(
                all_space_x, all_space_y, bins=[self.x_edges, self.y_edges]
            )
            results["toa_flux_map_2d"] = space_map
        else:
            results["toa_flux_map_2d"] = np.zeros((len(self.x_edges) - 1, len(self.y_edges) - 1))

        return results
