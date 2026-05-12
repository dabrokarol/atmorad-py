import numpy as np

from tqdm import tqdm

from src.physics import orientation, rotate, sun_elevation_rad_to_direction
from src.scene import Scene
from src.results import Results
from src.config import SimConfig

class MCRadiation:
    def __init__(self, config: SimConfig, scene: Scene, measure_z):

        self.num_photons = config.num_photons
        self.num_track = min(config.num_track, self.num_photons)
        self.starting_pos = np.array(config.starting_pos, dtype=np.float64).reshape(-1, 1)
        self.rng = np.random.default_rng(config.random_seed)
        
        self.theta_sun = config.theta_sun_deg
        self.phi_sun = config.phi_sun_deg

        self.results = None
        self.scene = scene

        self.measure_z = np.array(measure_z)
        self.measure_z[self.measure_z==0] = 1e-5 # move the z=0 detector infinitesimally downwards
        self.diff_down = np.zeros(measure_z.size + 1)
        self.diff_up = np.zeros(measure_z.size + 1)

    def _init_arrays(self):
        theta_sun_rad = self.theta_sun / 180 * np.pi
        phi_sun_rad = self.phi_sun / 180 * np.pi
        theta = self.rng.normal(theta_sun_rad, 1/60, size=(self.num_photons))
        phi = self.rng.normal(phi_sun_rad, 1/60, size=(self.num_photons))

        direction = sun_elevation_rad_to_direction(theta, phi)

        pos_x = self.rng.uniform(-1, 1, self.num_photons) * 100 + self.starting_pos[0]
        pos_y = self.rng.uniform(-1, 1, self.num_photons) * 100 + self.starting_pos[1]
        pos_z = np.full(self.num_photons, 1e-6)
        pos = np.vstack((pos_x, pos_y, pos_z))

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

        pbar = tqdm(total=self.num_photons, desc="Absorbed / total photons")

        while active_ids.size:
            num_active_photons = active_ids.size

            # Updating progress bar
            pbar.n = self.num_photons - num_active_photons
            pbar.refresh()

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

        pbar.close()

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
        )
    
    def get_results(self):
        if self.results is None:
            raise KeyError("No results, use '.run()' first")
        
        return self.results



