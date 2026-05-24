import numpy as np

from atmorad.config import SimConfig
from atmorad.constants import X, Y, Z
from atmorad.environment import Scene
from atmorad.models import IncidentFluxMapResult, PhotonBatch
from atmorad.registry import register_detector

from .base import BaseDetector


@register_detector("plane_flux", IncidentFluxMapResult)
class IncidentFluxMapDetector(BaseDetector):
    def __init__(self, scene: Scene, config: SimConfig):
        resolution = config.detectors.horizontal_maps_resolution_km

        self.measure_z = np.array(config.detectors.flux_maps_z_levels_km, dtype=float)
        self.domain_x = config.environment.geometry.domain_size_x_km
        self.domain_y = config.environment.geometry.domain_size_y_km

        self.num_bins_x = int(np.round(self.domain_x / resolution))
        self.num_bins_y = int(np.round(self.domain_y / resolution))
        self.num_planes = len(self.measure_z)

        self.dx = self.domain_x / self.num_bins_x
        self.dy = self.domain_y / self.num_bins_y

        self.x_edges = np.linspace(-self.domain_x / 2, self.domain_x / 2, self.num_bins_x + 1)
        self.y_edges = np.linspace(-self.domain_y / 2, self.domain_y / 2, self.num_bins_y + 1)

        self.total_bins = self.num_planes * self.num_bins_x * self.num_bins_y

        self.flux_down_flat = np.zeros(self.total_bins, dtype=np.float64)
        self.flux_up_flat = np.zeros(self.total_bins, dtype=np.float64)

    def _process_hits(
        self,
        batch: PhotonBatch,
        old_pos: np.ndarray,
        crossed_mask: np.ndarray,
        accumulator_flat: np.ndarray,
    ):
        if not np.any(crossed_mask):
            return

        photon_idx, plane_idx = np.where(crossed_mask)

        crossed_old_z = old_pos[Z, photon_idx]
        crossed_dir_z = batch.direction[Z, photon_idx]
        crossed_target_z = self.measure_z[plane_idx]

        t = (crossed_target_z - crossed_old_z) / crossed_dir_z

        exact_x = old_pos[X, photon_idx] + batch.direction[X, photon_idx] * t
        exact_y = old_pos[Y, photon_idx] + batch.direction[Y, photon_idx] * t

        wrapped_x = np.mod(exact_x + self.domain_x / 2, self.domain_x)
        wrapped_y = np.mod(exact_y + self.domain_y / 2, self.domain_y)

        idx_x = (wrapped_x / self.dx).astype(np.int64)
        idx_y = (wrapped_y / self.dy).astype(np.int64)

        idx_x = np.clip(idx_x, 0, self.num_bins_x - 1)
        idx_y = np.clip(idx_y, 0, self.num_bins_y - 1)

        flat_idx = plane_idx * (self.num_bins_x * self.num_bins_y) + idx_x * self.num_bins_y + idx_y

        weights = batch.weight[photon_idx]

        counts = np.bincount(flat_idx, weights=weights, minlength=self.total_bins)
        accumulator_flat += counts

    def record_movement(self, batch: PhotonBatch, old_pos: np.ndarray):
        old_z = old_pos[Z]
        new_z = batch.pos[Z]

        down_mask = (old_z[:, np.newaxis] > self.measure_z) & (
            new_z[:, np.newaxis] <= self.measure_z
        )
        up_mask = (old_z[:, np.newaxis] < self.measure_z) & (new_z[:, np.newaxis] >= self.measure_z)

        self._process_hits(batch, old_pos, down_mask, self.flux_down_flat)
        self._process_hits(batch, old_pos, up_mask, self.flux_up_flat)

    def get_results(self) -> IncidentFluxMapResult:
        x_centers = (self.x_edges[:-1] + self.x_edges[1:]) / 2.0
        y_centers = (self.y_edges[:-1] + self.y_edges[1:]) / 2.0

        flux_down_3d = self.flux_down_flat.reshape(
            (self.num_planes, self.num_bins_x, self.num_bins_y)
        )
        flux_up_3d = self.flux_up_flat.reshape((self.num_planes, self.num_bins_x, self.num_bins_y))

        return IncidentFluxMapResult(
            x_centers=x_centers,
            y_centers=y_centers,
            measure_z=self.measure_z,
            incident_flux_down_3d=flux_down_3d,
            incident_flux_up_3d=flux_up_3d,
        )
