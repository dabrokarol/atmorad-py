from abc import ABC, abstractmethod

import numpy as np

from atmorad.config import SimConfig
from atmorad.environment import Scene
from atmorad.models import PhotonBatch


class BaseDetector(ABC):
    @abstractmethod
    def __init__(self, scene: Scene, config: SimConfig):
        """Called by the Engine before the simulation starts."""

    @abstractmethod
    def record_movement(self, batch: PhotonBatch, old_pos: np.ndarray): ...

    @abstractmethod
    def record_interaction(
        self,
        batch: PhotonBatch,
        old_direction: np.ndarray,
        old_weight: np.ndarray,
        scatter_mask: np.ndarray,
        surface_mask: np.ndarray,
    ): ...

    @abstractmethod
    def record_termination(self, batch: PhotonBatch, terminated_mask: np.ndarray): ...

    @abstractmethod
    def finalize(self): ...

    @abstractmethod
    def get_results(self): ...
