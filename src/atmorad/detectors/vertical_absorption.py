import numpy as np

from atmorad.config import SimConfig
from atmorad.constants import Z
from atmorad.environment import Scene
from atmorad.models import AbsorptionProfileResult, PhotonBatch
from atmorad.registry import register_detector

from .base import BaseDetector


@register_detector("absorption_vertical", AbsorptionProfileResult)
class AbsorptionProfileDetector(BaseDetector):
    def __init__(self):
        self.scene: Scene | None = None
        self.spacing: float | None = None
        self.measure_z: np.ndarray | None = None

        self.absorption_profile: np.ndarray | None = None

    def initialize(self, scene: Scene, config: SimConfig):
        self.scene = scene
        top_of_atmosphere = scene.atmosphere.top_of_atmosphere
        self.spacing = config.detectors.vertical_profiles_resolution_km

        num_bins = int(np.round(top_of_atmosphere / self.spacing))
        self.measure_z = np.linspace(0, top_of_atmosphere, num_bins + 1)

        self.absorption_profile = np.zeros(num_bins, dtype=np.float64)

    def record_termination(self, batch: PhotonBatch, terminated_mask: np.ndarray):
        if not np.any(terminated_mask):
            return

        term_pos = batch.pos[:, terminated_mask]

        in_atmosphere_mask = self.scene.in_atmosphere(term_pos)

        if np.any(in_atmosphere_mask):
            absorbed_z = term_pos[Z, in_atmosphere_mask]

            term_w = batch.weight[terminated_mask]
            absorbed_w = term_w[in_atmosphere_mask]

            layer_indices = (absorbed_z / self.spacing).astype(np.int64)
            layer_indices = np.clip(layer_indices, 0, len(self.absorption_profile) - 1)

            layer_counts = np.bincount(
                layer_indices, weights=absorbed_w, minlength=len(self.absorption_profile)
            )
            self.absorption_profile += layer_counts

    def record_movement(self, batch: PhotonBatch, old_pos: np.ndarray):
        pass

    def record_scattering(
        self, batch: PhotonBatch, old_direction: np.ndarray, scattered_mask: np.ndarray
    ):
        pass

    def finalize(self):
        pass

    def get_results(self) -> AbsorptionProfileResult:
        z_centers = (self.measure_z[:-1] + self.measure_z[1:]) / 2.0

        return AbsorptionProfileResult(
            z_centers=z_centers, absorption_profile_1d=self.absorption_profile
        )
