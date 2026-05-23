import numpy as np

from atmorad.config import SimConfig
from atmorad.environment import Scene
from atmorad.models import FateResult, PhotonBatch
from atmorad.registry import register_detector

from .base import BaseDetector


@register_detector("fate", FateResult)
class FateDetector(BaseDetector):
    def __init__(self, scene: Scene, config: SimConfig):
        self.absorbed_surface = 0.0
        self.absorbed_atmosphere = 0.0
        self.escaped_toa = 0.0
        self.scene = scene

    def record_movement(self, batch: PhotonBatch, old_pos: np.ndarray):
        pass

    def record_scattering(
        self, batch: PhotonBatch, old_direction: np.ndarray, scattered_mask: np.ndarray
    ):
        pass

    def record_termination(self, batch: PhotonBatch, terminated_mask: np.ndarray):
        if not np.any(terminated_mask):
            return

        term_pos = batch.pos[:, terminated_mask]
        term_weight = batch.weight[terminated_mask]

        escaped_toa_mask = self.scene.above_toa(term_pos)
        absorbed_surface_mask = self.scene.below_ground(term_pos)
        absorbed_atmosphere_mask = ~escaped_toa_mask & ~absorbed_surface_mask

        self.escaped_toa += np.sum(term_weight[escaped_toa_mask])
        self.absorbed_surface += np.sum(term_weight[absorbed_surface_mask])
        self.absorbed_atmosphere += np.sum(term_weight[absorbed_atmosphere_mask])

    def finalize(self):
        pass

    def get_results(self) -> FateResult:
        return FateResult(
            photons_absorbed_surface=self.absorbed_surface,
            photons_absorbed_atmosphere=self.absorbed_atmosphere,
            photons_escaped_toa=self.escaped_toa,
        )
