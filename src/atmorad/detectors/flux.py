import numpy as np
from .base import BaseDetector 
from atmorad.environment import Scene
from atmorad.engine.batch import PhotonBatch
from atmorad.config import SimConfig      
from atmorad.constants import DETECTOR_OFFSET, Z

class VerticalFluxDetector(BaseDetector):
    def __init__(self):
        self.spacing = None
        self.measure_z = None
        self.diff_down = None
        self.diff_up = None

    def initialize(self, scene: Scene, config: SimConfig):
        self.scene = scene
        top_of_atmosphere = scene.atmosphere.get_total_thickness()
        self.spacing = config.detectors.vertical_flux_resolution_km

        self.measure_z = np.arange(0, top_of_atmosphere, self.spacing)
        self.measure_z[self.measure_z == 0] += DETECTOR_OFFSET 
        self.measure_z = np.append(self.measure_z, top_of_atmosphere - DETECTOR_OFFSET)
        
        self.diff_down = np.zeros(self.measure_z.size + 1)
        self.diff_up = np.zeros(self.measure_z.size + 1)

    def record_movement(self, batch: PhotonBatch, old_pos: np.ndarray):
        old_z = old_pos[Z]
        new_z = batch.pos[Z]

        down_mask = new_z < old_z
        if np.any(down_mask):
            z1_down = new_z[down_mask]
            z2_down = old_z[down_mask]
            idx_start = np.searchsorted(self.measure_z, z1_down, side='left')
            idx_end = np.searchsorted(self.measure_z, z2_down, side='right')
            start_bins = np.bincount(idx_start)
            end_bins = np.bincount(idx_end)
            self.diff_down[0:start_bins.size] += start_bins
            self.diff_down[0:end_bins.size] -= end_bins

        up_mask = new_z > old_z
        if np.any(up_mask):
            z1_up = old_z[up_mask] 
            z2_up = new_z[up_mask]
            idx_start = np.searchsorted(self.measure_z, z1_up, side='left')
            idx_end = np.searchsorted(self.measure_z, z2_up, side='right')
            start_bins = np.bincount(idx_start)
            end_bins = np.bincount(idx_end)
            self.diff_up[0:start_bins.size] += start_bins
            self.diff_up[0:end_bins.size] -= end_bins

    def get_results(self) -> dict:
        flux_down = np.cumsum(self.diff_down)[:-1]
        flux_up = np.cumsum(self.diff_up)[:-1]
        return {
            "measure_z": self.measure_z,
            "flux_up": flux_up,
            "flux_down": flux_down
        }