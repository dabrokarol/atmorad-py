import time
import logging
import numpy as np

from atmorad.physics import sun_zenith_to_direction
from atmorad.engine.batch import PhotonBatch
from atmorad.environment.scene import Scene
from atmorad.detectors.results import Results
from atmorad.config.config import SimConfig
from atmorad.constants import DETECTOR_OFFSET, EPSILON, MAX_SCATTERINGS, X, Y, Z

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
class Engine:
    def __init__(self, config: SimConfig, scene: Scene):
        self.num_photons = config.num_photons
        self.num_track = min(config.num_track, self.num_photons)
        self.rng = np.random.default_rng(config.random_seed)
        
        self.theta_sun = config.theta_sun_deg
        self.phi_sun = config.phi_sun_deg

        self.results = None
        self.scene = scene
        self.top_of_atmosphere = scene.top_of_atmosphere

        self.measure_z = np.arange(0, self.scene.atmosphere.get_total_thickness(), config.flux_measure_spacing, dtype=float)
        self.measure_z[self.measure_z == 0] += DETECTOR_OFFSET # move the z=0 detector infinitesimally upwards
        self.measure_z = np.append(self.measure_z, self.top_of_atmosphere - DETECTOR_OFFSET) # move the z=toa detector infinitesimally downwards
        self.diff_down = np.zeros(self.measure_z.size + 1)
        self.diff_up = np.zeros(self.measure_z.size + 1)

    def _init_arrays(self):
        theta_sun_rad = self.theta_sun / 180 * np.pi
        phi_sun_rad = self.phi_sun / 180 * np.pi
        theta = self.rng.normal(theta_sun_rad, 1/60, size=(self.num_photons))
        phi = self.rng.normal(phi_sun_rad, 1/60, size=(self.num_photons))
        direction = sun_zenith_to_direction(theta, phi)

        pos = np.empty(shape=(3, self.num_photons), dtype=float)
        pos[X, :] = self.rng.uniform(-1, 1, self.num_photons) * 100
        pos[Y, :] = self.rng.uniform(-1, 1, self.num_photons) * 100
        pos[Z, :] = np.full(self.num_photons, self.top_of_atmosphere - EPSILON)
        
        rand_component = self.rng.uniform(0, 1, self.num_photons)
        initial_material_ids = self.scene.atmosphere.get_mediums(pos, rand_component)
        
        batch = PhotonBatch(
            pos=pos,
            direction=direction,
            tau_to_travel=self.random_tau(self.num_photons),
            is_active=np.ones(self.num_photons, dtype=bool),
            ids=np.arange(self.num_photons),
            material_ids=initial_material_ids,
            scatter_counts=np.zeros(self.num_photons, dtype=int)
        )

        self.tracked_paths = {i: [] for i in range(self.num_track)}
        self.final_positions = np.zeros((3, self.num_photons))
        self.final_directions = np.zeros((3, self.num_photons))
        self.final_scatter_counts = np.zeros(self.num_photons, dtype=int)

        return batch

    def update_flux_counts(self, old_z: np.ndarray, new_z: np.ndarray):
        down_mask = new_z < old_z
        if np.any(down_mask):
            z1_down = new_z[down_mask]
            z2_down = old_z[down_mask]
            
            idx_start = np.searchsorted(self.measure_z, z1_down, side='left')
            idx_end = np.searchsorted(self.measure_z, z2_down, side='right')
            start_bins = np.bincount(idx_start)
            end_bins = np.bincount(idx_end)

            self.diff_down[0:start_bins.size] += start_bins
            self.diff_down[0:end_bins.size] -= end_bins

        up_mask = new_z > old_z
        if np.any(up_mask):
            z1_up = old_z[up_mask] 
            z2_up = new_z[up_mask]
            
            idx_start = np.searchsorted(self.measure_z, z1_up, side='left')
            idx_end = np.searchsorted(self.measure_z, z2_up, side='right')
            start_bins = np.bincount(idx_start)
            end_bins = np.bincount(idx_end)
            
            self.diff_up[0:start_bins.size] += start_bins
            self.diff_up[0:end_bins.size] -= end_bins

    def random_tau(self, size):
        return self.rng.exponential(scale=1.0, size=size)

    def run(self):
        batch = self._init_arrays()

        surface_hits_pos = []
        num_track = self.num_track
        scene = self.scene
        rng = self.rng
        
        start_time = time.process_time()

        while batch.active_count > 0:
            logging.info(f"Active photons: {batch.active_count}")
            
            for i, position in zip(batch.ids[batch.ids<num_track], batch.pos[:, batch.ids<num_track].copy().T):
                self.tracked_paths[i].append(position)

            tau_to_boundary = scene.tau_to_boundary(batch)

            new_tau_to_travel = np.where(tau_to_boundary < batch.tau_to_travel, tau_to_boundary, batch.tau_to_travel)
  
            old_pos_z = batch.pos[Z].copy()
            
            batch = scene.move_photons(batch, new_tau_to_travel)
            self.update_flux_counts(old_pos_z, batch.pos[Z])

            batch.tau_to_travel -= new_tau_to_travel
            scattering_event_mask = np.isclose(batch.tau_to_travel, 0)

            in_atmosphere_mask = self.scene.in_atmosphere(batch.pos)
            new_layer_mask = ~scattering_event_mask & in_atmosphere_mask
            rand_component = rng.uniform(0, 1, np.count_nonzero(new_layer_mask))
            batch.material_ids[new_layer_mask] = self.scene.atmosphere.get_mediums(batch.pos[:, new_layer_mask], rand_component)
            
            random_sample = self.rng.uniform(0, 1, size=(3, batch.active_count))
            batch, absorbed_surface, absorbed_atmosphere, scattered = scene.process_interactions(batch, scattering_event_mask, random_sample)
            batch.deactivate_photons(absorbed_surface | absorbed_atmosphere)

            batch.scatter_counts[scattered] += 1
            exceeded_scatterings_mask = batch.scatter_counts > MAX_SCATTERINGS
            
            new_tau_rand = self.random_tau(np.count_nonzero(scattered))
            batch.tau_to_travel[scattered] = new_tau_rand

            active_mask = ~self.scene.reached_space(batch.pos) & ~absorbed_surface & ~absorbed_atmosphere & ~exceeded_scatterings_mask
            terminated_mask = ~active_mask
            
            # Appending simulation results
            self.final_positions[:, batch.ids[terminated_mask]] = batch.pos[:, terminated_mask]
            self.final_directions[:, batch.ids[terminated_mask]] = batch.direction[:, terminated_mask]
            self.final_scatter_counts[batch.ids[terminated_mask]] = batch.scatter_counts[terminated_mask]
            
            if np.any(self.scene.reached_surface(batch.pos)):
                surface_hits_pos.append(batch.pos[:2, self.scene.reached_surface(batch.pos)].copy())
            
            batch.deactivate_photons(terminated_mask)
            batch.shrink_to_active()

        end_time = time.process_time()

        for i, position in enumerate(self.final_positions[:, :self.num_track].T):
            self.tracked_paths[i].append(position.copy())

        if surface_hits_pos:
            surface_hits_pos = np.concatenate(surface_hits_pos, axis=1)
        else:
            surface_hits_pos = np.array([])

        space_mask, surface_mask, layer_idx = scene.get_final_photon_position_data(self.final_positions)

        flux_down = np.cumsum(self.diff_down)[:-1]
        flux_up = np.cumsum(self.diff_up)[:-1]

        self.results = Results(
            final_positions=self.final_positions,
            space_mask=space_mask,
            surface_mask=surface_mask,
            layer_idx=layer_idx,
            sample_paths=self.tracked_paths,
            surface_hits=surface_hits_pos,
            scatter_counts=self.final_scatter_counts,
            measure_z=self.measure_z,
            flux_up=flux_up,
            flux_down=flux_down,
            cpu_time_s=(end_time - start_time) 
        )
    
    def get_results(self):
        if self.results is None:
            raise RuntimeError("No results, use '.run()' first")
        
        return self.results



