        

import numpy as np

from atmorad.environment.scene import Scene
from atmorad.engine.batch import PhotonBatch
from atmorad.config.config import SimConfig
from atmorad.detectors.base import BaseDetector

class FateDetector(BaseDetector):
    def __init__(self):
        self.absorbed_by_surface = None
        self.absorbed_by_atmosphere = None
        self.reached_space_mask = None
        self.scene = None
    
    def initialize(self, scene: Scene, config: SimConfig):
        self.absorbed_by_surface = 0
        self.absorbed_by_atmosphere = 0
        self.reached_space_mask = 0
        self.scene = scene
        
    def record_movement(self, batch: PhotonBatch, old_pos: np.ndarray):
        pass
        
    def record_termination(self, batch: PhotonBatch, terminated_mask: np.ndarray):
        reached_space_mask = self.scene.reached_space(batch.pos) & terminated_mask
        absorbed_surface_mask = self.scene.reached_surface(batch.pos) & terminated_mask
        absorbed_atmosphere_mask = ~reached_space_mask & ~absorbed_surface_mask & terminated_mask
        
        self.reached_space_mask += np.sum(reached_space_mask) 
        self.absorbed_by_surface += np.sum(absorbed_surface_mask)
        self.absorbed_by_atmosphere += np.sum(absorbed_atmosphere_mask)
        
    def finalize(self): ...
    
    def get_results(self) -> dict:
        return {
            "photons_absorbed_surface": self.absorbed_by_surface,
            "photons_absorbed_atmosphere": self.absorbed_by_atmosphere,
            "photons_reflected_toa": self.reached_space_mask,
        }
    
