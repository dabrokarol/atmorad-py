from typing import Self

import numpy as np
from pydantic import ConfigDict

from atmorad.config import SimConfig
from atmorad.constants import Z
from atmorad.environment import Scene
from atmorad.models import BaseResult, PhotonBatch
from atmorad.registry import register_detector

from .base import BaseDetector


class AbsorptionProfileResult(BaseResult):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    measure_z: np.ndarray
    absorption_profile_1d: np.ndarray

    def merge(self, other: Self) -> Self:
        return self.__class__(
            measure_z=self.measure_z,
            absorption_profile_1d=self.absorption_profile_1d + other.absorption_profile_1d,
        )


@register_detector("absorption_vertical")
class AbsorptionProfileDetector(BaseDetector):
    def __init__(self):
        self.spacing = None
        self.measure_z = None
        self.absorption_profile = None

    def initialize(self, scene: Scene, config: SimConfig):
        top_of_atmosphere = scene.atmosphere.get_total_thickness()
        self.spacing = config.detectors.vertical_profiles_resolution_km

        self.measure_z = np.arange(0, top_of_atmosphere, self.spacing)
        self.measure_z = np.append(self.measure_z, top_of_atmosphere)

        self.absorption_profile = np.zeros(self.measure_z.size - 1, dtype=np.int64)

    def record_termination(self, batch: PhotonBatch, terminated_mask: np.ndarray):
        if not np.any(terminated_mask):
            return

        term_pos = batch.pos[:, terminated_mask]
        toa = self.measure_z[-1]

        in_atmosphere_mask = (term_pos[Z] > 0.0) & (term_pos[Z] < toa)

        if np.any(in_atmosphere_mask):
            absorbed_z = term_pos[Z, in_atmosphere_mask]

            layer_indices = (absorbed_z / self.spacing).astype(np.int64)

            layer_indices = np.clip(layer_indices, 0, len(self.absorption_profile) - 1)

            layer_counts = np.bincount(layer_indices, minlength=len(self.absorption_profile))
            self.absorption_profile += layer_counts

    def record_movement(self, batch: PhotonBatch, old_pos: np.ndarray): ...

    def record_scattering(
        self, batch: PhotonBatch, old_direction: np.ndarray, scattered_mask: np.ndarray
    ): ...

    def finalize(self): ...

    def get_results(self) -> AbsorptionProfileResult:
        return AbsorptionProfileResult(
            measure_z=self.measure_z, absorption_profile_1d=self.absorption_profile
        )
