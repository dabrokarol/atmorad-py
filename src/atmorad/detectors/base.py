from abc import ABC, abstractmethod

import numpy as np

from atmorad.config import SimConfig
from atmorad.environment import Scene
from atmorad.models import BaseResult, PhotonBatch


class BaseDetector(ABC):
    """Abstract base class for detectors, defining Monte Carlo simulation lifecycle hooks."""

    @abstractmethod
    def __init__(self, scene: Scene, config: SimConfig):
        """Initializes the detector before the simulation starts."""
        pass

    def record_movement(self, batch: PhotonBatch):
        """Hook called after photon movement, before physical interactions."""
        pass

    def record_interaction(
        self,
        batch: PhotonBatch,
        old_direction: np.ndarray,
        old_weight: np.ndarray,
        scatter_mask: np.ndarray,
        surface_mask: np.ndarray,
    ):
        """Hook called after scattering or reflection to calculate deposited energy."""
        pass

    def record_termination(self, batch: PhotonBatch, terminated_mask: np.ndarray):
        """Hook called when photons are permanently deactivated (e.g., escaped, roulette)."""
        pass

    def finalize(self):
        """Hook called once at the end of the simulation loop for final calculations."""
        pass

    @abstractmethod
    def get_results(self) -> BaseResult:
        """Retrieves the final processed results from the detector."""
        pass
