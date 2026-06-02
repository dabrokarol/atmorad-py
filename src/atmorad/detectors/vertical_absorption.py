import numpy as np
import xarray as xr

from atmorad.config.schemas import SimConfig
from atmorad.environment import Scene
from atmorad.physics.batch import PhotonBatch

from .base import BaseDetector


class AbsorptionProfileDetector(BaseDetector):
    def __init__(self, scene: Scene, config: SimConfig):
        self.scene = scene
        top_of_atmosphere = scene.atmosphere.top_of_atmosphere
        self.spacing = config.detectors.absorption_profile.vertical_resolution_km

        self.measure_z = np.arange(0, top_of_atmosphere, self.spacing)
        if not np.isclose(self.measure_z[-1], top_of_atmosphere):
            self.measure_z = np.append(self.measure_z, top_of_atmosphere)

        num_bins = len(self.measure_z) - 1
        self.absorption_profile = np.zeros(num_bins, dtype=np.float64)

    def record_interaction(
        self,
        batch: PhotonBatch,
        scatter_mask: np.ndarray,
        surface_mask: np.ndarray,
    ):
        if not np.any(scatter_mask):
            return

        deposited_energy = batch.old_weight[scatter_mask] - batch.weight[scatter_mask]
        hit_z = batch.pos[2, scatter_mask]

        layer_indices = (hit_z / self.spacing).astype(np.int64)
        layer_indices = np.clip(layer_indices, 0, len(self.absorption_profile) - 1)

        layer_counts = np.bincount(
            layer_indices, weights=deposited_energy, minlength=len(self.absorption_profile)
        )
        self.absorption_profile += layer_counts

    def get_results(self) -> xr.Dataset:
        z_centers = (self.measure_z[:-1] + self.measure_z[1:]) / 2.0

        return xr.Dataset(
            data_vars={
                "absorption_profile_1d": (
                    ["z"],
                    self.absorption_profile,
                    {"units": "photons", "long_name": "Absorption Profile"},
                ),
            },
            coords={
                "z": ("z", z_centers, {"units": "km", "long_name": "Altitude"}),
            },
        )

    @staticmethod
    def merge_chunks(chunks: list[xr.Dataset]) -> xr.Dataset:
        if not chunks:
            return xr.Dataset()

        combined = xr.concat(chunks, dim="batch")
        return combined.sum(dim="batch", keep_attrs=True)
