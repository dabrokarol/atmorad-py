import numpy as np
import xarray as xr

from atmorad.config.schemas import SimConfig
from atmorad.environment import Scene
from atmorad.physics.batch import PhotonBatch

from .base import BaseDetector


class PathTrackingDetector(BaseDetector):
    def __init__(self, scene: Scene, config: SimConfig):
        self.num_track = min(
            config.detectors.trajectories.max_tracked_paths, config.engine.num_photons
        )
        self.tracked_paths = {i: [] for i in range(self.num_track)}
        self.tracked_weights = {i: [] for i in range(self.num_track)}
        self.scene = scene
        self.toa_z = self.scene.atmosphere.top_of_atmosphere

    def record_movement(self, batch: PhotonBatch):
        tracked_mask = batch.ids < self.num_track

        if np.any(tracked_mask):
            tracked_ids = batch.ids[tracked_mask]
            tracked_pos = batch.old_pos[:, tracked_mask]
            tracked_w = batch.weight[tracked_mask]

            for i, pos, w in zip(tracked_ids, tracked_pos.T, tracked_w):
                self.tracked_paths[i].append(pos.T.copy())
                self.tracked_weights[i].append(w)

    def record_termination(self, batch: PhotonBatch, terminated_mask: np.ndarray):
        tracked_term_mask = (batch.ids < self.num_track) & terminated_mask

        if np.any(tracked_term_mask):
            term_ids = batch.ids[tracked_term_mask]
            term_pos = batch.pos[:, tracked_term_mask]
            term_w = batch.weight[tracked_term_mask]

            for i, pos, w in zip(term_ids, term_pos.T, term_w):
                self.tracked_paths[i].append(pos.copy())
                self.tracked_weights[i].append(w)

    def get_results(self) -> xr.Dataset:
        if self.num_track == 0 or not self.tracked_paths:
            return xr.Dataset(attrs={"toa_z_km": self.toa_z})

        max_bounces = max(len(path) for path in self.tracked_paths.values())

        paths_3d = np.full((self.num_track, max_bounces, 3), np.nan)
        weights = np.full((self.num_track, max_bounces), np.nan)

        reflected = np.zeros(self.num_track, dtype=bool)
        abs_atm = np.zeros(self.num_track, dtype=bool)
        abs_surf = np.zeros(self.num_track, dtype=bool)

        for i in range(self.num_track):
            path = self.tracked_paths[i]
            w = self.tracked_weights[i]
            bounces = len(path)
            if bounces > 0:
                paths_3d[i, :bounces, :] = np.vstack(path)
                weights[i, :bounces] = w

                last_pos = path[-1]
                reflected[i] = self.scene.above_toa(last_pos.reshape(3, 1))[0]
                abs_atm[i] = self.scene.in_atmosphere(last_pos.reshape(3, 1))[0]
                abs_surf[i] = self.scene.at_surface(last_pos.reshape(3, 1))[0]

        return xr.Dataset(
            data_vars={
                "paths": (
                    ["photon", "path_step", "cartesian_axis"],
                    paths_3d,
                    {"units": "km", "long_name": "Photon path coordinates"},
                ),
                "weights": (
                    ["photon", "path_step"],
                    weights,
                    {"units": "1", "long_name": "Photon statistical weight"},
                ),
                "escaped_toa": (
                    ["photon"],
                    reflected,
                    {"units": "1", "long_name": "Reflected at top of atmosphere flag"},
                ),
                "absorbed_atmosphere": (
                    ["photon"],
                    abs_atm,
                    {"units": "1", "long_name": "Absorbed in atmosphere flag"},
                ),
                "absorbed_surface": (
                    ["photon"],
                    abs_surf,
                    {"units": "1", "long_name": "Absorbed at surface flag"},
                ),
            },
            coords={
                "cartesian_axis": (
                    "cartesian_axis",
                    ["x", "y", "z"],
                    {"long_name": "Cartesian axis"},
                ),
            },
            attrs={"toa_z_km": self.toa_z},
        )

    @staticmethod
    def merge_chunks(chunks: list[xr.Dataset]) -> xr.Dataset:
        valid_chunks = [c for c in chunks if "paths" in c.data_vars]
        if not valid_chunks:
            return xr.Dataset()

        return xr.concat(valid_chunks, dim="photon", join="outer", combine_attrs="no_conflicts")
