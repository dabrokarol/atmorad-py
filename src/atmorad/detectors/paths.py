import numpy as np

from atmorad.config import SimConfig
from atmorad.environment import Scene
from atmorad.models import PathTrackingResult, PhotonBatch
from atmorad.registry import register_detector

from .base import BaseDetector


@register_detector("path_tracking", PathTrackingResult)
class PathTrackingDetector(BaseDetector):
    def __init__(self, scene: Scene, config: SimConfig):
        self.num_track = min(config.detectors.num_full_paths, config.engine.num_photons)
        self.tracked_paths = {i: [] for i in range(self.num_track)}
        self.tracked_weights = {i: [] for i in range(self.num_track)}
        self.scene = scene
        self.toa_z = self.scene.atmosphere.top_of_atmosphere

    def record_movement(self, batch: PhotonBatch, old_pos: np.ndarray):
        tracked_mask = batch.ids < self.num_track

        if np.any(tracked_mask):
            tracked_ids = batch.ids[tracked_mask]
            tracked_pos = old_pos[:, tracked_mask]
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

    def get_results(self) -> PathTrackingResult:
        if self.num_track == 0 or not self.tracked_paths:
            return PathTrackingResult(
                sample_paths_3d=np.array([]),
                sample_weights_2d=np.array([]),
                sample_escaped_toa=np.array([]),
                sample_absorbed_atmosphere=np.array([]),
                sample_absorbed_surface=np.array([]),
                toa_z=self.toa_z,
            )

        max_bounces = max(len(path) for path in self.tracked_paths.values())

        paths_3d = np.full((self.num_track, max_bounces, 3), np.nan)
        weights_2d = np.full((self.num_track, max_bounces), np.nan)

        escaped = np.zeros(self.num_track, dtype=bool)
        abs_atm = np.zeros(self.num_track, dtype=bool)
        abs_surf = np.zeros(self.num_track, dtype=bool)

        for i in range(self.num_track):
            path = self.tracked_paths[i]
            weights = self.tracked_weights[i]
            bounces = len(path)
            if bounces > 0:
                paths_3d[i, :bounces, :] = np.vstack(path)
                weights_2d[i, :bounces] = weights

                last_pos = path[-1]
                escaped[i] = self.scene.above_toa(last_pos.reshape(3, 1))[0]
                abs_atm[i] = self.scene.in_atmosphere(last_pos.reshape(3, 1))[0]
                abs_surf[i] = self.scene.at_surface(last_pos.reshape(3, 1))[0]

        return PathTrackingResult(
            sample_paths_3d=paths_3d,
            sample_weights_2d=weights_2d,
            sample_escaped_toa=escaped,
            sample_absorbed_atmosphere=abs_atm,
            sample_absorbed_surface=abs_surf,
            toa_z=self.toa_z,
        )
