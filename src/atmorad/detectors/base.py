from abc import ABC

import numpy as np

from atmorad.config import SimConfig
from atmorad.models import PhotonBatch
from atmorad.environment import Scene


class BaseDetector(ABC):
    def initialize(self, scene: Scene, config: SimConfig):
        """Called by the Engine before the simulation starts."""

    def record_movement(self, batch: PhotonBatch, old_pos: np.ndarray): ...

    def record_scattering(
        self, batch: PhotonBatch, old_direction: np.ndarray, scattered_mask: np.ndarray
    ): ...

    def record_termination(self, batch: PhotonBatch, terminated_mask: np.ndarray): ...

    def finalize(self): ...

    def get_results(self) -> dict: ...
