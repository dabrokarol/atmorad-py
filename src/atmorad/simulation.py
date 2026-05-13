import time
import concurrent.futures
import numpy as np

from atmorad.physics import orientation, rotate, sun_elevation_rad_to_direction
from atmorad.scene import Scene
from atmorad.results import Results
from atmorad.config import SimConfig

EPSILON_DETECTOR = 1e-5
EPSILON_POS = 1e-10


class MCRadiation:
    def __init__(self, config: SimConfig, scene: Scene):
        self.config = config
        self.scene = scene

    def run(self):
        results = parallel_simulation(self.config, self.scene)
        self.results = Results.merge_all(results)

    def get_results(self):
        return self.results


def parallel_simulation(config: SimConfig, scene: Scene):
    chunk_size = config.num_photons // config.num_cores
    remainder = config.num_photons % config.num_cores
    seeds = np.random.SeedSequence(config.random_seed).spawn(config.num_cores)

    futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=config.num_cores) as executor:
        for i in range(config.num_cores):
            future = executor.submit(run_chunk, chunk_size + (remainder if i==0 else 0), seeds[i], config, scene, i)
            futures.append(future)

    all_results = [future.result() for future in concurrent.futures.as_completed(futures)]

    return all_results
        

def run_chunk(chunk_size: int, seed, config: SimConfig, scene: Scene, i):
    chunk_config = SimConfig(
        num_photons=chunk_size,
        num_track=config.num_track if i == 0 else 0,
        starting_pos=config.starting_pos,
        random_seed=seed,
        theta_sun_deg=config.theta_sun_deg,
        phi_sun_deg=config.phi_sun_deg,
        flux_measure_spacing=config.flux_measure_spacing
    )
    sim = Simulation(chunk_config, scene)
    sim.run()
    return sim.get_results()


class Simulation:
    def __init__(self, config: SimConfig, scene: Scene):
        self.num_photons = config.num_photons
        self.num_track = min(config.num_track, self.num_photons)
        self.starting_pos = np.array(config.starting_pos, dtype=np.float64).reshape(-1, 1)
        self.rng = np.random.default_rng(config.random_seed)
        
        self.theta_sun = config.theta_sun_deg
        self.phi_sun = config.phi_sun_deg

        self.results = None
        self.scene = scene

        self.measure_z = np.arange(0, self.scene.atmosphere.get_total_thickness(), config.flux_measure_spacing)
        self.measure_z[self.measure_z==0] = EPSILON_DETECTOR # move the z=0 detector infinitesimally downwards
        self.diff_down = np.zeros(self.measure_z.size + 1)
        self.diff_up = np.zeros(self.measure_z.size + 1)

    def _init_arrays(self):
        theta_sun_rad = self.theta_sun / 180 * np.pi
        phi_sun_rad = self.phi_sun / 180 * np.pi
        theta = self.rng.normal(theta_sun_rad, 1/60, size=(self.num_photons))
        phi = self.rng.normal(phi_sun_rad, 1/60, size=(self.num_photons))

        direction = sun_elevation_rad_to_direction(theta, phi)

        pos = np.empty(shape=(3, self.num_photons))
        pos[0, :] = self.rng.uniform(-1, 1, self.num_photons) * 100 + self.starting_pos[0]
        pos[1, :] = self.rng.uniform(-1, 1, self.num_photons) * 100 + self.starting_pos[1]
        pos[2, :] = np.full(self.num_photons, EPSILON_POS)

        ids = np.arange(0, self.num_photons)
        scatter_counts = np.zeros(self.num_photons)

        tracked_paths = {i: [] for i in range(self.num_track)}
        final_positions = np.zeros((3, self.num_photons))
        final_directions = np.zeros((3, self.num_photons))

        return pos, direction, ids, scatter_counts, tracked_paths, final_positions, final_directions

    def update_flux_counts(self, old_z: np.ndarray, new_z: np.ndarray):
        down_mask = new_z > old_z
        if np.any(down_mask):
            z1_down = old_z[down_mask]
            z2_down = new_z[down_mask]
            
            idx_start = np.searchsorted(self.measure_z, z1_down, side='right')
            idx_end = np.searchsorted(self.measure_z, z2_down, side='right')
            start_bins = np.bincount(idx_start)
            end_bins = np.bincount(idx_end)

            self.diff_down[0:start_bins.size] += start_bins
            self.diff_down[0:end_bins.size] -= end_bins

        up_mask = new_z < old_z
        if np.any(up_mask):
            z1_up = new_z[up_mask] 
            z2_up = old_z[up_mask]
            
            idx_start = np.searchsorted(self.measure_z, z1_up, side='left')
            idx_end = np.searchsorted(self.measure_z, z2_up, side='left')
            start_bins = np.bincount(idx_start)
            end_bins = np.bincount(idx_end)
            
            self.diff_up[0:start_bins.size] += start_bins
            self.diff_up[0:end_bins.size] -= end_bins

    def run(self):
        pos, direction, active_ids, scatter_counts, tracked_paths, final_positions, final_directions = self._init_arrays()

        surface_hits_pos = []
        
        num_track = self.num_track
        scene = self.scene

        start_time = time.perf_counter_ns()

        while active_ids.size:
            num_active_photons = active_ids.size

            for i, position in zip(active_ids[active_ids<num_track], pos[:, active_ids<num_track].copy().T):
                tracked_paths[i].append(position)

            transmission = self.rng.uniform(0, 1, num_active_photons)
            tau_to_travel = -np.log(transmission)

            old_z = pos[2].copy()
            pos, surface_mask, space_mask, medium_ids = scene.move_photons(pos, direction, tau_to_travel, self.rng)
            new_z = pos[2]
            self.update_flux_counts(old_z, new_z)
            
            atmosphere_mask = (~surface_mask) & (~space_mask)
            
            rand_interaction, rand_theta, rand_phi = self.rng.uniform(0, 1, size=(3, active_ids.size))
            direction, absorbed_surface, absorbed_atmosphere, scattered = scene.scatter_photons(pos, direction, rand_interaction, rand_theta, rand_phi, surface_mask, atmosphere_mask, medium_ids)

            active_mask = (~space_mask) & (~absorbed_surface) & ~(absorbed_atmosphere)
            terminated_mask = ~active_mask
            
            # Appending simulation results
            final_positions[:, active_ids[terminated_mask]] = pos[:, terminated_mask]
            final_directions[:, active_ids[terminated_mask]] = direction[:, terminated_mask]
            scatter_counts[active_ids[scattered]] += 1
            if np.any(surface_mask):
                surface_hits_pos.append(pos[:2, surface_mask].copy())
            
            # Shrinking arrays to alive photons
            pos = pos[:, active_mask]
            direction = direction[:, active_mask]
            active_ids = active_ids[active_mask]

        end_time = time.perf_counter_ns()

        for i, position in enumerate(final_positions[:, :self.num_track].T):
            tracked_paths[i].append(position.copy())

        if surface_hits_pos:
            surface_hits_pos = np.concatenate(surface_hits_pos, axis=1)
        else:
            surface_hits_pos = np.array([])

        space_mask, surface_mask, layer_idx = scene.get_photon_position_mask(final_positions[2])

        flux_down = np.cumsum(self.diff_down)[:-1]
        flux_up = np.cumsum(self.diff_up)[:-1]

        self.results = Results(
            final_positions=final_positions,
            space_mask=space_mask,
            surface_mask=surface_mask,
            layer_idx=layer_idx,
            sample_paths=tracked_paths,
            surface_hits=surface_hits_pos,
            scatter_counts=scatter_counts,
            measure_z=self.measure_z,
            flux_up=flux_up,
            flux_down=flux_down,
            sim_duration_s=(end_time - start_time) / 1e9 
        )
    
    def get_results(self):
        if self.results is None:
            raise RuntimeError("No results, use '.run()' first")
        
        return self.results



