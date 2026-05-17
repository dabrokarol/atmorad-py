import numpy as np
from atmorad.detectors.base import BaseDetector
from atmorad.environment.scene import Scene
from atmorad.engine.batch import PhotonBatch
from atmorad.config.config import SimConfig
from atmorad.constants import Z, EPSILON

class AtmosphericHeatingRateDetector(BaseDetector):
    def __init__(self):
        self.num_layers = None
        self.layer_boundaries = None
        self.absorption_profile = None

    def initialize(self, scene: Scene, config: SimConfig):
        self.num_layers = len(scene.atmosphere.boundaries) - 1
        self.layer_boundaries = scene.atmosphere.boundaries.copy()
        self.absorption_profile = np.zeros(self.num_layers, dtype=int)
        
    def record_termination(self, batch: PhotonBatch, terminated_mask: np.ndarray):
        if not np.any(terminated_mask):
            return

        term_pos = batch.pos[:, terminated_mask]
        toa = self.layer_boundaries[-1]
        in_atmosphere_mask = (term_pos[Z] > (EPSILON * 2)) & (term_pos[Z] < (toa - EPSILON * 2))
        
        if np.any(in_atmosphere_mask):
            absorbed_z = term_pos[Z, in_atmosphere_mask]
            layer_indices = np.searchsorted(self.layer_boundaries, absorbed_z, 'right') - 1
            layer_indices = np.clip(layer_indices, 0, self.num_layers - 1)
            
            layer_counts = np.bincount(layer_indices, minlength=self.num_layers)
            self.absorption_profile += layer_counts

    def get_results(self) -> dict:
        return {
            "heating_profile_1d": self.absorption_profile,
            "layer_boundaries_z": self.layer_boundaries
        }