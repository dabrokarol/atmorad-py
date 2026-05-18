import numpy as np
from .base import BaseDetector
from atmorad.environment import Scene
from atmorad.engine.batch import PhotonBatch
from atmorad.config import SimConfig

class PathTrackingDetector(BaseDetector):
    def __init__(self):
        self.num_track = 0
        self.tracked_paths = {}

    def initialize(self, scene: Scene, config: SimConfig):
        self.num_track = min(config.detectors.num_full_paths, config.engine.num_photons)
        self.tracked_paths = {i: [] for i in range(self.num_track)}

    def record_movement(self, batch: PhotonBatch, old_pos: np.ndarray):
        tracked_mask = batch.ids < self.num_track
        
        if np.any(tracked_mask):
            tracked_ids = batch.ids[tracked_mask]
            tracked_pos = old_pos[:, tracked_mask]
            
            for i, pos in zip(tracked_ids, tracked_pos.T):
                self.tracked_paths[i].append(pos.copy())

    def record_termination(self, batch: PhotonBatch, terminated_mask: np.ndarray):
        tracked_term_mask = (batch.ids < self.num_track) & terminated_mask
        
        if np.any(tracked_term_mask):
            term_ids = batch.ids[tracked_term_mask]
            term_pos = batch.pos[:, tracked_term_mask]
            
            for i, pos in zip(term_ids, term_pos.T):
                self.tracked_paths[i].append(pos.copy())

    def get_results(self) -> dict:
        return {
            "sample_paths": self.tracked_paths
        }