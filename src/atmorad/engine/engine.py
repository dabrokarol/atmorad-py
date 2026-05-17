import time
import logging
import numpy as np

from atmorad.engine.batch import PhotonBatch
from atmorad.environment.scene import Scene
from atmorad.detectors.base import BaseDetector
from atmorad.config.config import SimConfig
from atmorad.constants import MAX_SCATTERINGS, X, Y, Z
class Engine:
    def __init__(self, config: SimConfig, scene: Scene, detectors: list[BaseDetector]):
        self.config = config
        self.scene = scene
        self.detectors = detectors
        
        engine_config = config.engine
        source_config = config.source
        self.num_photons = engine_config.num_photons
        self.rng = np.random.default_rng(engine_config.random_seed)
        self.theta_sun = source_config.theta_sun_deg
        self.phi_sun = source_config.phi_sun_deg

        self.results = None

    def _init_arrays(self):        
        pos = self.scene.start_pos(self.num_photons, self.rng)
        direction = self.scene.start_direction(self.num_photons, self.theta_sun, self.phi_sun, self.rng)
        material_ids = self.scene.get_material_ids(pos, self.rng)
        
        batch = PhotonBatch(
            pos = pos,
            direction = direction,
            is_active=np.ones(self.num_photons, dtype=bool),
            tau_to_travel=self.random_tau(self.num_photons),
            ids=np.arange(self.num_photons),
            material_ids=material_ids,
            scatter_counts=np.zeros(self.num_photons, dtype=int),
        )

        return batch

    def random_tau(self, size):
        return self.rng.exponential(scale=1.0, size=size)

    def run(self):
        for det in self.detectors:
            det.initialize(self.scene, self.config)
        
        batch = self._init_arrays()

        scene = self.scene
        rng = self.rng
        
        start_time = time.process_time()

        while batch.active_count > 0:
            logging.debug(f"Active photons: {batch.active_count}")

            tau_to_boundary = scene.tau_to_boundary(batch)

            new_tau_to_travel = np.where(tau_to_boundary < batch.tau_to_travel, tau_to_boundary, batch.tau_to_travel)
  
            old_pos = batch.pos.copy()
            batch = scene.move_photons(batch, new_tau_to_travel)
            
            for det in self.detectors:
                det.record_movement(batch, old_pos)

            batch.tau_to_travel -= new_tau_to_travel
            scattering_event_mask = np.isclose(batch.tau_to_travel, 0)

            in_atmosphere_mask = self.scene.in_atmosphere(batch.pos)
            new_layer_mask = ~scattering_event_mask & in_atmosphere_mask
            batch.material_ids[new_layer_mask] = self.scene.get_material_ids(batch.pos[:, new_layer_mask], rng)
            
            random_sample = rng.uniform(0, 1, size=(3, batch.active_count))
            old_direction = batch.direction.copy()
            batch, absorbed_surface, absorbed_atmosphere, scattered = scene.process_interactions(batch, scattering_event_mask, random_sample)
            batch.deactivate_photons(absorbed_surface | absorbed_atmosphere)
            
            for det in self.detectors:
                det.record_scattering(batch, old_direction, scattered)

            batch.scatter_counts[scattered] += 1
            exceeded_scatterings_mask = batch.scatter_counts > MAX_SCATTERINGS
            logging.warning(f"Killing {np.count_nonzero(exceeded_scatterings_mask)} photons. Scattered more than {MAX_SCATTERINGS} times.")
            
            new_tau_rand = self.random_tau(np.count_nonzero(scattered))
            batch.tau_to_travel[scattered] = new_tau_rand

            active_mask = ~self.scene.reached_space(batch.pos) & ~absorbed_surface & ~absorbed_atmosphere & ~exceeded_scatterings_mask
            terminated_mask = ~active_mask
              
            for det in self.detectors:
                det.record_termination(batch, terminated_mask)
            
            batch.deactivate_photons(terminated_mask)
            batch.shrink_to_active()

        end_time = time.process_time()

        engine_results = {
            "cpu_time_s": (end_time - start_time)
        }

        detector_results = {}
        for det in self.detectors:
            detector_results.update(det.get_results())
        self.results = {**engine_results, **detector_results}
    
    def get_results(self):
        if self.results is None:
            raise RuntimeError("No results, use '.run()' first")
        
        return self.results