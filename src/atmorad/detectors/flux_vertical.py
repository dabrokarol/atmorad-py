import numpy as np

from atmorad.config import SimConfig
from atmorad.constants import Z
from atmorad.environment import Scene
from atmorad.models import PhotonBatch, VerticalFluxResult
from atmorad.registry import register_detector

from .base import BaseDetector


@register_detector("vertical_flux", VerticalFluxResult)
class VerticalFluxDetector(BaseDetector):
    def __init__(self, scene: Scene, config: SimConfig):
        self.scene = scene
        top_of_atmosphere = scene.atmosphere.top_of_atmosphere
        self.spacing = config.detectors.vertical_profiles_resolution_km

        self.measure_z = np.arange(0, top_of_atmosphere, self.spacing)
        if not np.isclose(self.measure_z[-1], top_of_atmosphere):
            self.measure_z = np.append(self.measure_z, top_of_atmosphere)

        self.diff_down = np.zeros(self.measure_z.size + 1, dtype=float)
        self.diff_up = np.zeros(self.measure_z.size + 1, dtype=float)

    def record_movement(self, batch: PhotonBatch, old_pos: np.ndarray):
        old_z = old_pos[Z]
        new_z = batch.pos[Z]
        weight = batch.weight
        max_idx = len(self.measure_z) + 1

        down_mask = new_z < old_z
        if np.any(down_mask):
            z_start = new_z[down_mask]
            z_end = old_z[down_mask]
            w_down = weight[down_mask]

            idx_start = np.ceil(z_start / self.spacing).astype(np.int64)
            idx_end = np.floor(z_end / self.spacing).astype(np.int64) + 1

            idx_start = np.clip(idx_start, 0, max_idx)
            idx_end = np.clip(idx_end, 0, max_idx)

            start_bins = np.bincount(idx_start, weights=w_down, minlength=max_idx)
            end_bins = np.bincount(idx_end, weights=w_down, minlength=max_idx)

            self.diff_down += start_bins
            self.diff_down -= end_bins

        up_mask = new_z > old_z
        if np.any(up_mask):
            z_start = old_z[up_mask]
            z_end = new_z[up_mask]
            w_up = weight[up_mask]

            idx_start = np.ceil(z_start / self.spacing).astype(np.int64)
            idx_end = np.floor(z_end / self.spacing).astype(np.int64) + 1

            idx_start = np.clip(idx_start, 0, max_idx)
            idx_end = np.clip(idx_end, 0, max_idx)

            start_bins = np.bincount(idx_start, weights=w_up, minlength=max_idx)
            end_bins = np.bincount(idx_end, weights=w_up, minlength=max_idx)

            self.diff_up += start_bins
            self.diff_up -= end_bins

    def get_results(self) -> VerticalFluxResult:
        flux_down = np.cumsum(self.diff_down)[:-1]
        flux_up = np.cumsum(self.diff_up)[:-1]

        return VerticalFluxResult(measure_z=self.measure_z, flux_up=flux_up, flux_down=flux_down)
