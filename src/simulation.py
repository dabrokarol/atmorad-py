import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from pathlib import Path

from .physics import orientation, rotate
from .scene import Scene

#### ASSUMPTIONS (TO EASE LATER)
# Isotropic atmoshperic layer (tau, albedo)
# Perfectly black surface layer (can be changed to brdf)
# Sun position constant (could be taken from some database)
#### OTHER TODOS
# Add consistent colors for plotting

class MCRadiation:
    def __init__(self, config, scene: Scene, rng):
        self.img_dir = Path.cwd() / config['filepaths']['img_dir']
        self.plot_name = config['filepaths']['plot_name']
        self.img_dir.mkdir(exist_ok=True)

        self.n_photons = config['general']['n_photons']
        self.n_track = config['general']['n_track']
        self.starting_pos = np.array(config['general']['starting_pos'], dtype=np.float64).reshape(-1, 1)
        
        self.theta_sun = config['sun']['theta_sun']
        self.phi_sun = config['sun']['phi_sun']

        self.results = None

        self.scene = scene
        self.rng = rng

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

        pos = np.tile(self.starting_pos, (1, self.n_photons))
        ids = np.arange(0, self.n_photons)
        scatter_counts = np.zeros(self.n_photons)

        history = {i: [] for i in range(self.n_track)}
        last_positions = np.zeros((self.n_photons, 3))

        return pos, ori, ids, scatter_counts, history, last_positions

    def run(self):
        pos, ori, ids, scatter_counts, history, last_positions = self._init_arrays()
        
        n_track = self.n_track
        scene = self.scene

        n_left_atmosphere = 0
        n_absorbed_surf = 0
        n_absorbed_atmoshpere = 0

        while ids.size:
            n_photons = ids.size

            logging.info(f"{n_photons} photons left")
            for i, position in zip(ids[ids<n_track], pos[:, ids<n_track].T.copy()):
                history[i].append(position)

            transmission = self.rng.uniform(0, 1, n_photons)
            tau_to_travel = -np.log(transmission)

            pos, surface_mask, space_mask, medium_ids = scene.move_photons(pos, ori, tau_to_travel, self.rng)
            
            atmoshpere_mask = (~surface_mask) & (~space_mask)
            
            r1, r2, r3 = self.rng.uniform(0, 1, size=(3, ids.size))
            ori, absorbed_surface, absorbed_atmosphere = scene.scatter_photons(pos, ori, r1, r2, r3, surface_mask, atmoshpere_mask, medium_ids)


            n_left_atmosphere += np.count_nonzero(space_mask)
            n_absorbed_atmoshpere += np.count_nonzero(absorbed_atmosphere)
            n_absorbed_surf += np.count_nonzero(absorbed_surface)

            msk = (~space_mask) & (~absorbed_surface) & ~(absorbed_atmosphere)
            left_msk = ~msk
            
            last_positions[ids[left_msk]] = pos[:, left_msk].T
            
            #### SHRINKING THE ARRAY TO ONLY ALIVE PHOTONS
            pos = pos[:, msk]
            ori = ori[:, msk]
            ids = ids[msk]

        for i, positon in enumerate(last_positions[:self.n_track]):
            history[i].append(positon.copy())


        self.results = {
            "photons left atmosphere": n_left_atmosphere,
            "photons absorbed by surface": n_absorbed_surf,
            "photons absorbed by atmosphere": n_absorbed_atmoshpere,
            "sample paths": history
        }

    def plot_paths(self, name:str|None=None):
        if name is None:
            name = self.plot_name

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        if self.results is not None:
            history = self.results['sample paths']
        else:
            raise KeyError("No photon paths found. Use '.run()' first")

        starting = []
        ending = []
        lines = []
        for i, h in history.items(): 
            X, Y, Z = np.array(h).T
            p1 = ax.scatter(X[0], Y[0], Z[0], color='green', label='starting-point', alpha=0.7, s=5)
            l1, = ax.plot(X, Y, Z, label=f'{i}', alpha=0.7)
            p2 = ax.scatter(X[-1], Y[-1], Z[-1], color='red', label='ending-point', alpha=0.7, s=5)
            starting.append(p1)
            lines.append(l1)
            ending.append(p2)

        ax.invert_zaxis()
        # source: https://stackoverflow.com/questions/31478077/how-to-make-two-markers-share-the-same-label-in-the-legend
        fig.legend([tuple(starting), tuple(lines), tuple(ending)], ['start points', 'paths', 'ending points'], handler_map={tuple: HandlerTuple(ndivide=None)})
        fig.savefig(self.img_dir / name)

    def print_results(self):
        if self.results is not None:
            results = self.results
        else:
            raise KeyError("No results found. Use '.run()' first")

        for k, v in results.items():
            print(f"{k}: {v}") if k != "sample paths" else ...

