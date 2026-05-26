import logging
import time

import numpy as np

from atmorad.config import SimConfig
from atmorad.constants import EPSILON, MAX_SCATTERINGS
from atmorad.detectors import BaseDetector
from atmorad.environment.scene import Scene
from atmorad.models import EngineResult, PhotonBatch, SimResults
from atmorad.registry import DETECTORS


class Engine:
    def __init__(self, config: SimConfig, scene: Scene):
        self.config = config
        self.scene = scene

        engine_config = config.engine
        source_config = config.source
        self.num_photons = engine_config.num_photons
        self.rng = np.random.default_rng(engine_config.random_seed)
        self.theta_sun = source_config.theta_sun_deg
        self.phi_sun = source_config.phi_sun_deg
        self.weight_threshold = config.engine.photon_weight_threshold
        self.survival_chance = config.engine.photon_survival_chance
        self.weight_multiplier = 1.0 / config.engine.photon_survival_chance

        self.results = None

    def _init_arrays(self):
        pos = self.scene.start_pos(self.num_photons, self.rng)
        direction = self.scene.start_direction(self.num_photons, self.theta_sun, self.phi_sun)
        material_ids = self.scene.get_material_ids(pos, self.rng)

        batch = PhotonBatch(
            pos=pos,
            direction=direction,
            weight=np.ones(self.num_photons, dtype=float),
            is_active=np.ones(self.num_photons, dtype=bool),
            tau_to_travel=self.random_tau(self.num_photons),
            ids=np.arange(self.num_photons),
            material_ids=material_ids,
            scatter_counts=np.zeros(self.num_photons, dtype=int),
        )

        return batch

    def random_tau(self, size):
        return self.rng.exponential(scale=1.0, size=size)

    def _initialize_detectors(self):
        self.detectors: dict[str, BaseDetector] = {}

        for det_name in self.config.detectors.active:
            detector_class = DETECTORS[det_name]
            self.detectors[det_name] = detector_class(self.scene, self.config)

    def run(self):
        old_err = np.seterr(divide="ignore", invalid="ignore")

        self._initialize_detectors()
        batch = self._init_arrays()

        scene = self.scene
        rng = self.rng

        start_time = time.process_time()

        while batch.active_count > 0:
            logging.debug(f"Active photons: {batch.active_count}")

            batch.update_old_pos()

            batch, tau_consumed = scene.move_photons(batch)

            batch.tau_to_travel -= tau_consumed

            for det in self.detectors.values():
                det.record_movement(batch)

            scatter_mask = batch.tau_to_travel <= EPSILON
            surface_mask = self.scene.at_surface(batch.pos)
            in_atmosphere_mask = self.scene.in_atmosphere(batch.pos)

            new_layer_mask = ~scatter_mask & in_atmosphere_mask
            batch.material_ids[new_layer_mask] = self.scene.get_material_ids(
                batch.pos[:, new_layer_mask], rng
            )

            old_direction = batch.direction.copy()
            old_weight = batch.weight.copy()

            batch = scene.process_interactions(batch, scatter_mask, surface_mask, rng)

            for det in self.detectors.values():
                det.record_interaction(
                    batch,
                    old_direction,
                    old_weight,
                    scatter_mask,
                    surface_mask,
                )

            batch.scatter_counts[scatter_mask] += 1
            exceeded_scatterings_mask = batch.scatter_counts > MAX_SCATTERINGS
            if exceeded_scatterings_mask.any():
                logging.warning(
                    "Killing %d photons. Scattered more than %d times.",
                    np.count_nonzero(exceeded_scatterings_mask),
                    MAX_SCATTERINGS,
                )

            new_tau_rand = self.random_tau(np.count_nonzero(scatter_mask))
            batch.tau_to_travel[scatter_mask] = new_tau_rand

            low_weight_mask = batch.weight < self.weight_threshold
            killed_by_roulette = np.zeros(batch.active_count, dtype=bool)

            if np.any(low_weight_mask):
                num_low = np.count_nonzero(low_weight_mask)
                survive_rolls = rng.random(num_low)

                survivors_submask = survive_rolls < self.survival_chance

                new_weights = np.zeros(num_low)
                new_weights[survivors_submask] = (
                    batch.weight[low_weight_mask][survivors_submask] * self.weight_multiplier
                )

                died_submask = ~survivors_submask
                killed_by_roulette[low_weight_mask] = died_submask

                batch.weight[low_weight_mask] = new_weights

            reflected_toa = self.scene.above_toa(batch.pos)

            terminated_mask = reflected_toa | killed_by_roulette | exceeded_scatterings_mask

            for det in self.detectors.values():
                det.record_termination(batch, terminated_mask)

            batch.deactivate_photons(terminated_mask)
            batch.shrink_to_active()

        np.seterr(**old_err)

        end_time = time.process_time()

        self.cpu_time_s = end_time - start_time
        self.results = self._build_results()

    def _build_results(self) -> SimResults:
        detector_results = {}

        for det_id, det in self.detectors.items():
            det.finalize()
            detector_results[det_id] = det.get_results()

        return SimResults(
            engine_result=EngineResult(
                cpu_time_s=self.cpu_time_s,
            ),
            detector_results=detector_results,
            total_photons=self.num_photons,
        )

    def get_results(self):
        if self.results is None:
            raise RuntimeError("No results, use '.run()' first")

        return self.results
