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

        self.n_photons = config['general']['n_photons']
        self.n_track = config['general']['n_track']
        self.starting_pos = np.array(config['general']['starting_pos'], dtype=np.float64).reshape(-1, 1)
        self.rng = np.random.default_rng(config['general']['random_seed'])
        
        self.theta_sun = config['sun']['theta_sun']
        self.phi_sun = config['sun']['phi_sun']

        self.results = None
        self.scene = scene

    def _init_arrays(self):
        theta_sun_rad = self.theta_sun / 180 * np.pi
        phi_sun_rad = self.phi_sun / 180 * np.pi
        theta = self.rng.normal(theta_sun_rad, 1/60, size=(self.n_photons))
        phi = self.rng.normal(phi_sun_rad, 1/60, size=(self.n_photons))

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        cos_p = np.cos(phi)
        sin_p = np.sin(phi)

        ori = orientation(cos_t, sin_t, cos_p, sin_p)

        pos_x = self.rng.uniform(-1, 1, self.n_photons) * 100 + self.starting_pos[0]
        pos_y = self.rng.uniform(-1, 1, self.n_photons) * 100 + self.starting_pos[1]
        pos_z = np.full(self.n_photons, 1e-6)
        pos = np.vstack((pos_x, pos_y, pos_z))

        ids = np.arange(0, self.n_photons)
        scatter_counts = np.zeros(self.n_photons)

        history = {i: [] for i in range(self.n_track)}
        last_positions = np.zeros((3, self.n_photons))

        return pos, ori, ids, scatter_counts, history, last_positions

    def run(self):
        pos, ori, ids, scatter_counts, history, last_positions = self._init_arrays()
        
        n_track = self.n_track
        scene = self.scene

        while ids.size:
            n_photons = ids.size

            logging.info(f"{n_photons} photons left")
            for i, position in zip(ids[ids<n_track], pos[:, ids<n_track].copy().T):
                history[i].append(position)

            transmission = self.rng.uniform(0, 1, n_photons)
            tau_to_travel = -np.log(transmission)

            pos, surface_mask, space_mask, medium_ids = scene.move_photons(pos, ori, tau_to_travel, self.rng)
            
            atmoshpere_mask = (~surface_mask) & (~space_mask)
            
            r1, r2, r3 = self.rng.uniform(0, 1, size=(3, ids.size))
            ori, absorbed_surface, absorbed_atmosphere = scene.scatter_photons(pos, ori, r1, r2, r3, surface_mask, atmoshpere_mask, medium_ids)

            msk = (~space_mask) & (~absorbed_surface) & ~(absorbed_atmosphere)
            left_msk = ~msk
            
            last_positions[:, ids[left_msk]] = pos[:, left_msk]
            
            #### SHRINKING THE ARRAY TO ONLY ALIVE PHOTONS
            pos = pos[:, msk]
            ori = ori[:, msk]
            ids = ids[msk]

        for i, positon in enumerate(last_positions[:, :self.n_track].T):
            history[i].append(positon.copy())

        space_mask, surface_mask, layer_idx = scene.get_photon_position_msk(last_positions[2])

        self.results = Results(
            last_positions=last_positions,
            space_mask=space_mask,
            surface_mask=surface_mask,
            layer_idx=layer_idx,
            sample_paths=history
        )
    
    def get_results(self):
        if self.results is None:
            raise KeyError("No results, use '.run()' first")
        
        return self.results



