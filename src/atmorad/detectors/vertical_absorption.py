import numpy as np

from atmorad.config import SimConfig
from atmorad.environment import Scene
from atmorad.models import AbsorptionProfileResult, PhotonBatch
from atmorad.registry import register_detector

from .base import BaseDetector


@register_detector("absorption_vertical", AbsorptionProfileResult)
class AbsorptionProfileDetector(BaseDetector):
    def __init__(self, scene: Scene, config: SimConfig):
        self.scene = scene
        top_of_atmosphere = scene.atmosphere.top_of_atmosphere
        self.spacing = config.detectors.vertical_profiles_resolution_km

        num_bins = int(np.round(top_of_atmosphere / self.spacing))
        self.measure_z = np.linspace(0, top_of_atmosphere, num_bins + 1)

        self.absorption_profile = np.zeros(num_bins, dtype=np.float64)

    def record_interaction(
        self,
        batch: PhotonBatch,
        old_direction: np.ndarray,
        old_weight: np.ndarray,
        scatter_mask: np.ndarray,
        surface_mask: np.ndarray,
    ):
        if not np.any(scatter_mask):
            return

        deposited_energy = old_weight[scatter_mask] - batch.weight[scatter_mask]
        hit_z = batch.pos[2, scatter_mask]

        layer_indices = (hit_z / self.spacing).astype(np.int64)
        layer_indices = np.clip(layer_indices, 0, len(self.absorption_profile) - 1)

        layer_counts = np.bincount(
            layer_indices, weights=deposited_energy, minlength=len(self.absorption_profile)
        )
        self.absorption_profile += layer_counts

    def get_results(self) -> AbsorptionProfileResult:
        z_centers = (self.measure_z[:-1] + self.measure_z[1:]) / 2.0

        return AbsorptionProfileResult(
            z_centers=z_centers, absorption_profile_1d=self.absorption_profile
        )
