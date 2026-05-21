import numpy as np

from atmorad.config import SimConfig
from atmorad.constants import Z
from atmorad.environment import Scene
from atmorad.models import PhotonBatch

from .base import BaseDetector


class VerticalFluxDetector(BaseDetector):
    def __init__(self):
        self.spacing = None
        self.measure_z = None
        self.diff_down = None
        self.diff_up = None

    def initialize(self, scene: Scene, config: SimConfig):
        self.scene = scene
        top_of_atmosphere = scene.atmosphere.get_total_thickness()
        self.spacing = config.detectors.vertical_profiles_resolution_km

        self.measure_z = np.arange(0, top_of_atmosphere, self.spacing)
        self.measure_z = np.append(self.measure_z, top_of_atmosphere)

        self.diff_down = np.zeros(self.measure_z.size + 1, dtype=int)
        self.diff_up = np.zeros(self.measure_z.size + 1, dtype=int)

    def record_movement(self, batch: PhotonBatch, old_pos: np.ndarray):
        old_z = old_pos[Z]
        new_z = batch.pos[Z]

        old_idx = (old_z / self.spacing).astype(np.int64)
        new_idx = (new_z / self.spacing).astype(np.int64)

        max_idx = len(self.measure_z) + 1
        old_idx = np.clip(old_idx, 0, max_idx)
        new_idx = np.clip(new_idx, 0, max_idx)

        down_mask = new_z < old_z
        if np.any(down_mask):
            z_start = new_z[down_mask]
            z_end = old_z[down_mask]

            idx_start = np.ceil(z_start / self.spacing).astype(np.int64)
            idx_end = np.floor(z_end / self.spacing).astype(np.int64) + 1

            idx_start = np.clip(idx_start, 0, max_idx)
            idx_end = np.clip(idx_end, 0, max_idx)

            start_bins = np.bincount(idx_start)
            end_bins = np.bincount(idx_end)

            self.diff_down[0 : start_bins.size] += start_bins
            self.diff_down[0 : end_bins.size] -= end_bins

        up_mask = new_z > old_z
        if np.any(up_mask):
            z_start = old_z[up_mask]
            z_end = new_z[up_mask]

            idx_start = np.ceil(z_start / self.spacing).astype(np.int64)
            idx_end = np.floor(z_end / self.spacing).astype(np.int64) + 1

            idx_start = np.clip(idx_start, 0, max_idx)
            idx_end = np.clip(idx_end, 0, max_idx)

            start_bins = np.bincount(idx_start)
            end_bins = np.bincount(idx_end)

            self.diff_up[0 : start_bins.size] += start_bins
            self.diff_up[0 : end_bins.size] -= end_bins

    def get_results(self) -> dict:
        flux_down = np.cumsum(self.diff_down)[:-1]
        flux_up = np.cumsum(self.diff_up)[:-1]

        return {"measure_z": self.measure_z, "flux_up": flux_up, "flux_down": flux_down}
