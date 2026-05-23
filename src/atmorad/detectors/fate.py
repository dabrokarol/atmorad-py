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

    def record_interaction(
        self, batch, old_direction, old_weight, surface_mask, scatter_mask
    ):
        if np.any(scatter_mask):
            deposited = old_weight[scatter_mask] - batch.weight[scatter_mask]
            self.absorbed_atmosphere += np.sum(deposited)

        if np.any(surface_mask):
            deposited = old_weight[surface_mask] - batch.weight[surface_mask]
            self.absorbed_surface += np.sum(deposited)

    def record_termination(self, batch, terminated_mask):
        if not np.any(terminated_mask):
            return

        term_pos = batch.pos[:, terminated_mask]
        term_weight = batch.weight[terminated_mask]

        escaped_toa_mask = self.scene.above_toa(term_pos)
        if np.any(escaped_toa_mask):
            self.escaped_toa += np.sum(term_weight[escaped_toa_mask])
    def finalize(self):
        pass

    def get_results(self) -> FateResult:
        return FateResult(
            photons_absorbed_surface=self.absorbed_surface,
            photons_absorbed_atmosphere=self.absorbed_atmosphere,
            photons_escaped_toa=self.escaped_toa,
        )
