import numpy as np
import xarray as xr

from atmorad.config import SimConfig
from atmorad.constants import X, Y
from atmorad.environment import Scene
from atmorad.physics.batch import PhotonBatch

from .base import BaseDetector


class SurfaceAbsorptionDetector(BaseDetector):
    def __init__(self, scene: Scene, config: SimConfig):
        self.scene = scene
        self.domain_x = config.environment.geometry.domain_size_x_km
        self.domain_y = config.environment.geometry.domain_size_y_km

        resolution = config.detectors.horizontal_maps_resolution_km
        num_bins_x = int(np.round(self.domain_x / resolution))
        num_bins_y = int(np.round(self.domain_y / resolution))

        self.x_edges = np.linspace(-self.domain_x / 2, self.domain_x / 2, num_bins_x + 1)
        self.y_edges = np.linspace(-self.domain_y / 2, self.domain_y / 2, num_bins_y + 1)

        self.surface_map = np.zeros((num_bins_x, num_bins_y), dtype=float)

    def record_interaction(
        self,
        batch: PhotonBatch,
        scatter_mask: np.ndarray,
        surface_mask: np.ndarray,
    ):
        if not np.any(surface_mask):
            return

        deposited_energy = batch.old_weight[surface_mask] - batch.weight[surface_mask]

        hit_x = batch.pos[X, surface_mask]
        hit_y = batch.pos[Y, surface_mask]

        wrapped_x = np.mod(hit_x + self.domain_x / 2, self.domain_x) - self.domain_x / 2
        wrapped_y = np.mod(hit_y + self.domain_y / 2, self.domain_y) - self.domain_y / 2

        batch_map, _, _ = np.histogram2d(
            wrapped_x, wrapped_y, bins=[self.x_edges, self.y_edges], weights=deposited_energy
        )

        self.surface_map += batch_map

    def get_results(self) -> xr.Dataset:
        x_centers = (self.x_edges[:-1] + self.x_edges[1:]) / 2.0
        y_centers = (self.y_edges[:-1] + self.y_edges[1:]) / 2.0

        return xr.Dataset(
            data_vars={
                "surface_absorption_map_2d": (
                    ["x", "y"],
                    self.surface_map,
                    {"units": "photons", "long_name": "Surface Absorption"},
                ),
            },
            coords={
                "x": ("x", x_centers, {"units": "km", "long_name": "X Coordinate"}),
                "y": ("y", y_centers, {"units": "km", "long_name": "Y Coordinate"}),
            },
        )

    @staticmethod
    def merge_chunks(chunks: list[xr.Dataset]) -> xr.Dataset:
        if not chunks:
            return xr.Dataset()

        combined = xr.concat(chunks, dim="batch")
        return combined.sum(dim="batch", keep_attrs=True)
