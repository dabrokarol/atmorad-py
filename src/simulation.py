import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from pathlib import Path

from .physics import orientation, rotate, hg_cos_theta
from .atmosphere import Atmosphere
from .boundaries import Surface, Space

#### ASSUMPTIONS (TO EASE LATER)
# Isotropic atmoshperic layer (tau, albedo)
# Perfectly black surface layer (can be changed to brdf)
# Sun position constant (could be taken from some database)
#### OTHER TODOS
# Add consistent colors for plotting

class MCRadiation:
    def __init__(self, config, atm: Atmosphere, sur: Surface, spa: Space):
        self.img_dir = Path.cwd() / config['filepaths']['img_dir']
        self.plot_name = config['filepaths']['plot_name']
        self.img_dir.mkdir(exist_ok=True)

        self.n_photons = config['general']['n_photons']
        self.n_track = config['general']['n_track']
        self.starting_pos = np.array(config['general']['starting_pos'], dtype=np.float64).reshape(-1, 1)
        
        self.theta_sun = config['sun']['theta_sun']
        self.phi_sun = config['sun']['phi_sun']

        self.results = None

        self.atmoshpere = atm
        self.surface = sur
        self.space = spa

    def _init_arrays(self):
        theta_sun_rad = self.theta_sun / 180 * np.pi
        phi_sun_rad = self.phi_sun / 180 * np.pi
        thetas = np.random.normal(theta_sun_rad, 1/60, size=(self.n_photons))
        phis = np.random.normal(phi_sun_rad, 1/60, size=(self.n_photons))
        ori = orientation(thetas, phis)

        pos = np.tile(self.starting_pos, (1, self.n_photons))
        ids = np.arange(0, self.n_photons)
        scatter_counts = np.zeros(self.n_photons)

        history = {i: [] for i in range(self.n_track)}
        last_positions = np.zeros((self.n_photons, 3))

        return pos, ori, ids, scatter_counts, history, last_positions

    def run(self):
        pos, ori, ids, scatter_counts, history, last_positions = self._init_arrays()
        
        n_track = self.n_track

        n_left_atmosphere = 0
        n_absorbed_surf = 0
        n_absorbed_atmoshpere = 0

        while ids.size:
            n_photons = ids.size
            transmission = np.random.uniform(0, 1, n_photons)

            for i, positon in zip(ids[ids < n_track], pos[:, ids < n_track].T):
                history[i].append(positon.copy())

            tau = -np.log(transmission)
            dist = self.atmoshpere.calc_dist(pos, ori, tau)

            pos += ori * dist

            reached_space = self.atmoshpere.check_reached_space(pos)
            reached_surf = self.atmoshpere.check_reached_surf(pos)
            pos = self.atmoshpere.snap_to_boundaries(pos, ori, reached_space, reached_surf)

            n_reached = np.count_nonzero(reached_surf)
            rand_surf = np.random.uniform(0, 1, n_reached)

            reflected_surf = np.zeros_like(reached_surf, dtype=bool)
            if n_reached > 0:
                reflected_surf[reached_surf] = self.surface.check_reflection(pos[:, reached_surf], rand_surf)

            absorbed_surf = ~reflected_surf & reached_surf

            rand_t = np.random.uniform(0, 1, np.count_nonzero(reflected_surf))
            rand_p = np.random.uniform(0, 1, np.count_nonzero(reflected_surf))
            ori[:, reflected_surf] = self.surface.reflect(pos[:, reflected_surf], ori[:, reflected_surf], rand_t, rand_p)
            
            last_positions[ids[reached_space]] = pos[:, reached_space].T
            last_positions[ids[absorbed_surf]] = pos[:, absorbed_surf].T

            n_left_atmosphere += np.count_nonzero(reached_space)
            n_absorbed_surf += np.count_nonzero(absorbed_surf)

            pos[2, reached_space] = np.inf
            pos[2, absorbed_surf] = np.inf

            to_scatter = (~reached_space) & (~reached_surf)
            w = np.random.uniform(0, 1, np.count_nonzero(to_scatter))

            scattered = np.zeros_like(to_scatter, dtype=bool)

            res_scat = self.atmoshpere.check_scat(pos[:, to_scatter], ori[:, to_scatter], w)
            scattered[to_scatter] = res_scat

            absorbed = ~scattered & to_scatter            

            last_positions[ids[absorbed]] = pos[:, absorbed].T
            pos[2, absorbed] = np.inf

            n_absorbed_atmoshpere += np.count_nonzero(absorbed)
            n_scattered = np.count_nonzero(scattered)
        
            rand_t = np.random.uniform(0, 1, n_scattered)
            rand_p = np.random.uniform(0, 1, n_scattered)

            cos_t, sin_t, cos_p, sin_p = self.atmoshpere.scatter(pos[:, scattered], ori[:, scattered], rand_t, rand_p)

            ori[:, scattered] = rotate(ori[:, scattered], cos_t, sin_t, cos_p, sin_p)
            
            scatter_counts[ids[scattered]] += 1
            
            #### SHRINKING THE ARRAY TO ONLY ALIVE PHOTONS
            msk = pos[2] != np.inf
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

