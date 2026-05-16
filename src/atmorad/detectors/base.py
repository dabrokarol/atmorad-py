from abc import ABC

import numpy as np

from atmorad.environment.atmosphere import Atmosphere
from atmorad.environment.scene import Scene
from atmorad.engine.batch import PhotonBatch
from atmorad.config.config import SimConfig

class BaseDetector(ABC):
    def initialize(self, scene: Scene, config: SimConfig):
        pass
        
    def record_movement(self, batch: PhotonBatch, old_pos: np.ndarray, new_pos: np.ndarray):
        pass
        
    def record_scattering(self, batch: PhotonBatch, scattered_mask: np.ndarray):
        pass
        
    def record_termination(self, batch: PhotonBatch, terminated_mask: np.ndarray, scene: Scene):
        pass
        
    def finalize(self):
        pass