import numpy as np

from atmorad.config import SimConfig
from atmorad.constants import X, Y, Z
from atmorad.environment import Scene
from atmorad.models import IncidentFluxMapResult, PhotonBatch
from atmorad.registry import register_detector

from .base import BaseDetector


@register_detector("plane_flux", IncidentFluxMapResult)
class IncidentFluxMapDetector(BaseDetector):
    def __init__(self):
        self.measure_z: np.ndarray | None = None
        self.domain_x: float | None = None
        self.domain_y: float | None = None

        self.x_edges: np.ndarray | None = None
        self.y_edges: np.ndarray | None = None
        self.p_edges: np.ndarray | None = None
        
        self.flux_down_3d: np.ndarray | None = None
        self.flux_up_3d: np.ndarray | None = None

    def initialize(self, scene: Scene, config: SimConfig):
        resolution = config.detectors.horizontal_maps_resolution_km
        
        self.measure_z = np.array(config.detectors.flux_maps_z_levels_km, dtype=float)
        self.domain_x = config.environment.geometry.domain_size_x_km
        self.domain_y = config.environment.geometry.domain_size_y_km
        
        num_bins_x = int(np.round(self.domain_x / resolution))
        num_bins_y = int(np.round(self.domain_y / resolution))
        num_planes = len(self.measure_z)
        
        self.x_edges = np.linspace(-self.domain_x / 2, self.domain_x / 2, num_bins_x + 1)
        self.y_edges = np.linspace(-self.domain_y / 2, self.domain_y / 2, num_bins_y + 1)

        self.p_edges = np.arange(num_planes + 1) - 0.5

        self.flux_down_3d = np.zeros((num_planes, num_bins_x, num_bins_y), dtype=np.float64)
        self.flux_up_3d = np.zeros((num_planes, num_bins_x, num_bins_y), dtype=np.float64)
    
    def _process_hits(
        self,
        batch: PhotonBatch,
        old_pos: np.ndarray,
        crossed_mask: np.ndarray,
        accumulator: np.ndarray,
    ):
        if not np.any(crossed_mask):
            return

        photon_idx, plane_idx = np.where(crossed_mask)

        crossed_old_z = old_pos[Z, photon_idx]
        crossed_dir_z = batch.direction[Z, photon_idx]
        crossed_target_z = self.measure_z[plane_idx]

        with np.errstate(divide="ignore", invalid="ignore"):
            t = (crossed_target_z - crossed_old_z) / crossed_dir_z

        exact_x = old_pos[X, photon_idx] + batch.direction[X, photon_idx] * t
        exact_y = old_pos[Y, photon_idx] + batch.direction[Y, photon_idx] * t

        wrapped_x = np.mod(exact_x + self.domain_x / 2, self.domain_x) - self.domain_x / 2
        wrapped_y = np.mod(exact_y + self.domain_y / 2, self.domain_y) - self.domain_y / 2

        weights = batch.weight[photon_idx]

        sample = np.column_stack((plane_idx, wrapped_x, wrapped_y))

        batch_hist, _ = np.histogramdd(
            sample, 
            bins=[self.p_edges, self.x_edges, self.y_edges],
            weights=weights
        )

        accumulator += batch_hist

    def record_movement(self, batch: PhotonBatch, old_pos: np.ndarray):
        old_z = old_pos[Z]
        new_z = batch.pos[Z]

        down_mask = (old_z[:, np.newaxis] > self.measure_z) & (new_z[:, np.newaxis] <= self.measure_z)
        up_mask = (old_z[:, np.newaxis] < self.measure_z) & (new_z[:, np.newaxis] >= self.measure_z)

        self._process_hits(batch, old_pos, down_mask, self.flux_down_3d)
        self._process_hits(batch, old_pos, up_mask, self.flux_up_3d)

    def record_scattering(
        self, batch: PhotonBatch, old_direction: np.ndarray, scattered_mask: np.ndarray
    ):
        pass

    def record_termination(self, batch: PhotonBatch, terminated_mask: np.ndarray):
        pass

    def finalize(self):
        pass

    def get_results(self) -> IncidentFluxMapResult:
        x_centers = (self.x_edges[:-1] + self.x_edges[1:]) / 2.0
        y_centers = (self.y_edges[:-1] + self.y_edges[1:]) / 2.0
        
        return IncidentFluxMapResult(
            x_centers=x_centers,
            y_centers=y_centers,
            measure_z=self.measure_z,
            incident_flux_down_3d=self.flux_down_3d,
            incident_flux_up_3d=self.flux_up_3d
        )
