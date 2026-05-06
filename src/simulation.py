import logging
import numpy as np

from src.physics import orientation, rotate
from src.scene import Scene
from src.results import Results

# TODO:
# Sun position constant (could be taken from some database)

class MCRadiation:
    def __init__(self, config, scene: Scene):
        config = config['simulation']

        self.num_photons = config['general']['n_photons']
        self.num_track = min(config['general']['n_track'], self.num_photons)
        self.starting_pos = np.array(config['general']['starting_pos'], dtype=np.float64).reshape(-1, 1)
        self.rng = np.random.default_rng(config['general']['random_seed'])
        
        self.theta_sun = config['sun']['theta_sun']
        self.phi_sun = config['sun']['phi_sun']

        self.results = None
        self.scene = scene

    def _init_arrays(self):
        theta_sun_rad = self.theta_sun / 180 * np.pi
        phi_sun_rad = self.phi_sun / 180 * np.pi
        theta = self.rng.normal(theta_sun_rad, 1/60, size=(self.num_photons))
        phi = self.rng.normal(phi_sun_rad, 1/60, size=(self.num_photons))

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        direction = orientation(cos_theta, sin_theta, cos_phi, sin_phi)

        pos_x = self.rng.uniform(-1, 1, self.num_photons) * 100 + self.starting_pos[0]
        pos_y = self.rng.uniform(-1, 1, self.num_photons) * 100 + self.starting_pos[1]
        pos_z = np.full(self.num_photons, 1e-6)
        pos = np.vstack((pos_x, pos_y, pos_z))

        ids = np.arange(0, self.num_photons)
        scatter_counts = np.zeros(self.num_photons)

        tracked_paths = {i: [] for i in range(self.num_track)}
        final_positions = np.zeros((3, self.num_photons))

        return pos, direction, ids, scatter_counts, tracked_paths, final_positions

    def run(self):
        pos, direction, active_ids, scatter_counts, tracked_paths, final_positions = self._init_arrays()
        
        num_track = self.num_track
        scene = self.scene

        while active_ids.size:
            num_active_photons = active_ids.size

            logging.info(f"{num_active_photons} photons left")
            for i, position in zip(active_ids[active_ids<num_track], pos[:, active_ids<num_track].copy().T):
                tracked_paths[i].append(position)

            transmission = self.rng.uniform(0, 1, num_active_photons)
            tau_to_travel = -np.log(transmission)

            pos, surface_mask, space_mask, medium_ids = scene.move_photons(pos, direction, tau_to_travel, self.rng)
            
            atmosphere_mask = (~surface_mask) & (~space_mask)
            
            rand_interaction, rand_theta, rand_phi = self.rng.uniform(0, 1, size=(3, active_ids.size))
            direction, absorbed_surface, absorbed_atmosphere = scene.scatter_photons(pos, direction, rand_interaction, rand_theta, rand_phi, surface_mask, atmosphere_mask, medium_ids)

            active_mask = (~space_mask) & (~absorbed_surface) & ~(absorbed_atmosphere)
            terminated_mask = ~active_mask
            
            final_positions[:, active_ids[terminated_mask]] = pos[:, terminated_mask]
            
            #### SHRINKING THE ARRAY TO ONLY ALIVE PHOTONS
            pos = pos[:, active_mask]
            direction = direction[:, active_mask]
            active_ids = active_ids[active_mask]

        for i, positon in enumerate(final_positions[:, :self.num_track].T):
            tracked_paths[i].append(positon.copy())

        space_mask, surface_mask, layer_idx = scene.get_photon_position_mask(final_positions[2])

        self.results = Results(
            final_positions=final_positions,
            space_mask=space_mask,
            surface_mask=surface_mask,
            layer_idx=layer_idx,
            sample_paths=tracked_paths
        )
    
    def get_results(self):
        if self.results is None:
            raise KeyError("No results, use '.run()' first")
        
        return self.results



