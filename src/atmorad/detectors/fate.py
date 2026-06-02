import numpy as np
import xarray as xr

from atmorad.config import SimConfig
from atmorad.environment import Scene
from atmorad.physics.batch import PhotonBatch

from .base import BaseDetector


class FateDetector(BaseDetector):
    def __init__(self, scene: Scene, config: SimConfig):
        self.absorbed_surface = 0.0
        self.absorbed_atmosphere = 0.0
        self.escaped_toa = 0.0
        self.scene = scene

    def record_interaction(
        self, batch: PhotonBatch, scatter_mask: np.ndarray, surface_mask: np.ndarray
    ):
        if np.any(scatter_mask):
            deposited = batch.old_weight[scatter_mask] - batch.weight[scatter_mask]
            self.absorbed_atmosphere += np.sum(deposited)

        if np.any(surface_mask):
            deposited = batch.old_weight[surface_mask] - batch.weight[surface_mask]
            self.absorbed_surface += np.sum(deposited)

    def record_termination(self, batch: PhotonBatch, terminated_mask: np.ndarray):
        if not np.any(terminated_mask):
            return

        term_pos = batch.pos[:, terminated_mask]
        term_weight = batch.weight[terminated_mask]

        escaped_toa_mask = self.scene.above_toa(term_pos)
        if np.any(escaped_toa_mask):
            self.escaped_toa += np.sum(term_weight[escaped_toa_mask])

    def get_results(self) -> xr.Dataset:
        return xr.Dataset(
            data_vars={
                "energy_absorbed_surface": (
                    [],
                    self.absorbed_surface,
                    {"units": "photons", "long_name": "Energy Absorbed at Surface"},
                ),
                "energy_absorbed_atmosphere": (
                    [],
                    self.absorbed_atmosphere,
                    {"units": "photons", "long_name": "Energy Absorbed in Atmosphere"},
                ),
                "energy_outgoing_toa": (
                    [],
                    self.escaped_toa,
                    {"units": "photons", "long_name": "Energy Escaped TOA"},
                ),
            }
        )

    @staticmethod
    def merge_chunks(chunks: list[xr.Dataset]) -> xr.Dataset:
        if not chunks:
            return xr.Dataset()

        combined = xr.concat(chunks, dim="batch")

        return combined.sum(dim="batch", keep_attrs=True)
