
import numpy as np

from atmorad.config import SimConfig
from atmorad.environment import Scene
from atmorad.models import FateResult, PhotonBatch
from atmorad.registry import register_detector

from .base import BaseDetector


@register_detector("fate")
class FateDetector(BaseDetector):
    def __init__(self):
        self.absorbed_surface = None
        self.absorbed_atmosphere = None
        self.above_toa_mask = None
        self.scene = None

    def initialize(self, scene: Scene, config: SimConfig):
        self.absorbed_surface = 0
        self.absorbed_atmosphere = 0
        self.above_toa_mask = 0
        self.scene = scene

    def record_movement(self, batch: PhotonBatch, old_pos: np.ndarray):
        pass

    def record_scattering(
        self, batch: PhotonBatch, old_direction: np.ndarray, scattered_mask: np.ndarray
    ):
        pass

    def record_termination(self, batch: PhotonBatch, terminated_mask: np.ndarray):
        escaped_toa_mask = self.scene.above_toa(batch.pos) & terminated_mask
        absorbed_surface_mask = self.scene.below_ground(batch.pos) & terminated_mask
        absorbed_atmosphere_mask = ~escaped_toa_mask & ~absorbed_surface_mask & terminated_mask

        self.above_toa_mask += np.sum(escaped_toa_mask)
        self.absorbed_surface += np.sum(absorbed_surface_mask)
        self.absorbed_atmosphere += np.sum(absorbed_atmosphere_mask)

    def finalize(self): ...

    def get_results(self) -> FateResult:
        return FateResult(
            photons_absorbed_surface=self.absorbed_surface,
            photons_absorbed_atmosphere=self.absorbed_atmosphere,
            photons_escaped_toa=self.above_toa_mask,
        )
