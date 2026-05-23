import logging
import time

import numpy as np

from atmorad.config import SimConfig
from atmorad.constants import EPSILON, MAX_SCATTERINGS
from atmorad.detectors import BaseDetector
from atmorad.environment.scene import Scene
from atmorad.models import EngineResult, PhotonBatch, SimulationResults
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
        self._initialize_detectors()
        batch = self._init_arrays()

        scene = self.scene
        rng = self.rng

        start_time = time.process_time()

        while batch.active_count > 0:
            logging.debug(f"Active photons: {batch.active_count}")

            tau_to_boundary = scene.tau_to_boundary(batch)

            new_tau_to_travel = np.where(
                tau_to_boundary < batch.tau_to_travel, tau_to_boundary, batch.tau_to_travel
            )

            old_pos = batch.pos.copy()
            batch = scene.move_photons(batch, new_tau_to_travel)

            for det in self.detectors.values():
                det.record_movement(batch, old_pos)

            batch.tau_to_travel -= new_tau_to_travel
            scattering_event_mask = np.isclose(batch.tau_to_travel, 0, atol=EPSILON)

            in_atmosphere_mask = self.scene.in_atmosphere(batch.pos)
            new_layer_mask = ~scattering_event_mask & in_atmosphere_mask
            batch.material_ids[new_layer_mask] = self.scene.get_material_ids(
                batch.pos[:, new_layer_mask], rng
            )

            random_sample = rng.uniform(0, 1, size=(3, batch.active_count))
            old_direction = batch.direction.copy()
            batch, absorbed_surface, absorbed_atmosphere, scattered = scene.process_interactions(
                batch, scattering_event_mask, random_sample
            )
            batch.deactivate_photons(absorbed_surface | absorbed_atmosphere)

            for det in self.detectors.values():
                det.record_scattering(batch, old_direction, scattered)

            batch.scatter_counts[scattered] += 1
            exceeded_scatterings_mask = batch.scatter_counts > MAX_SCATTERINGS
            if exceeded_scatterings_mask.any():
                logging.warning(
                    f"Killing {np.count_nonzero(exceeded_scatterings_mask)} photons. Scattered more than {MAX_SCATTERINGS} times."
                )

            new_tau_rand = self.random_tau(np.count_nonzero(scattered))
            batch.tau_to_travel[scattered] = new_tau_rand

            active_mask = (
                ~self.scene.above_toa(batch.pos)
                & ~absorbed_surface
                & ~absorbed_atmosphere
                & ~exceeded_scatterings_mask
            )
            terminated_mask = ~active_mask

            for det in self.detectors.values():
                det.record_termination(batch, terminated_mask)

            batch.deactivate_photons(terminated_mask)
            batch.shrink_to_active()

        end_time = time.process_time()

        self.cpu_time_s = end_time - start_time
        self.results = self._build_results()

    def _build_results(self) -> SimulationResults:
        detector_results = {}

        for det_id, det in self.detectors.items():
            det.finalize()
            detector_results[det_id] = det.get_results()

        return SimulationResults(
            engine=EngineResult(
                cpu_time_s=self.cpu_time_s,
            ),
            detector_results=detector_results,
        )

    def get_results(self):
        if self.results is None:
            raise RuntimeError("No results, use '.run()' first")

        return self.results
