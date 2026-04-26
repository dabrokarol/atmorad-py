import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from helpers import read_config, orientation, rotate, hg_cos_theta

#### ASSUMPTIONS (TO EASE LATER)
# Isotropic atmoshperic layer (tau, albedo)
# Perfectly black surface layer (can be changed to brdf)
# Sun position constant (could be taken from some database)
#### OTHER TODOS
# Add consistent colors for plotting

class MCRadiation:
    def __init__(self, config):
        self.img_dir = Path.cwd() / config['simulation']['filepaths']['img_dir']
        self.plot_name = config['simulation']['filepaths']['plot_name']
        self.img_dir.mkdir(exist_ok=True)

        self.n_photons = config['simulation']['general']['n_photons']
        self.n_track = config['simulation']['general']['n_track']
        self.starting_pos = np.array(config['simulation']['general']['starting_pos'])
        
        self.tau_star = config['simulation']['atmoshpere']['tau_star']
        self.ss_albedo = config['simulation']['atmoshpere']['omega']
        
        self.theta_sun = config['simulation']['sun']['theta_sun']
        self.phi_sun = config['simulation']['sun']['phi_sun']

        self.g = config['simulation']['scattering']['g']
        
        scat_type = config['simulation']['scattering']['type']
        if scat_type == 'henyey-greenstein':
            self.scat_func = hg_cos_theta
        else:
            raise KeyError(f"Unknown scattering func {scat_type}, check config")

        self.results = None

    def _init_arrays(self):
        theta_sun_rad = self.theta_sun / 180 * np.pi
        phi_sun_rad = self.phi_sun / 180 * np.pi
        thetas = np.random.normal(theta_sun_rad, 1/60, size=(self.n_photons))
        phis = np.random.normal(phi_sun_rad, 1/60, size=(self.n_photons))
        ori = orientation(thetas, phis)

        pos = np.zeros(shape=(3, self.n_photons), dtype=np.float64)
        ids = np.arange(0, self.n_photons)
        scatter_counts = np.zeros(self.n_photons)

        history = {i: [] for i in range(self.n_track)}
        last_positions = np.zeros((self.n_photons, 3))

        return pos, ori, ids, scatter_counts, history, last_positions

    def run(self):
        pos, ori, ids, scatter_counts, history, last_positions = self._init_arrays()
        
        n_track = self.n_track

        n_left_atmosphere = 0
        n_hit_ground = 0
        n_absorbed_atmoshpere = 0

        while ids.size:
            n_photons = ids.size
            transmission = np.random.uniform(0, 1, n_photons)

            for i, positon in zip(ids[ids < n_track], pos[:, ids < n_track].T):
                history[i].append(positon.copy())

            dist = -np.log(transmission)

            pos += ori * dist

            left = (pos[2] < 0)
            hit = (pos[2] > self.tau_star)

            last_pos_left = pos[:, left] + (0 - pos[2, left]) / ori[2, left] * ori[:, left]
            last_pos_hit = pos[:, hit] + (self.tau_star - pos[2, hit]) / ori[2, hit] * ori[:, hit]
            last_positions[ids[left]] = last_pos_left.T
            last_positions[ids[hit]] = last_pos_hit.T

            n_left_atmosphere += np.count_nonzero(left)
            n_hit_ground += np.count_nonzero(hit)

            pos[2, left] = np.inf
            pos[2, hit] = np.inf
            
            w = np.random.uniform(0, 1, n_photons)

            scattered = (w < self.ss_albedo) & (pos[2] < np.inf)
            absorbed = (w >= self.ss_albedo) & (pos[2] < np.inf)

            last_positions[ids[absorbed]] = pos[:, absorbed].T
            pos[2, absorbed] = np.inf

            n_absorbed_atmoshpere += np.count_nonzero(absorbed)
            n_scattered = np.count_nonzero(scattered)
        
            #### HEYNEY GREENSTEIN phase function
            cos_theta_prim = self.scat_func(np.random.uniform(0, 1, n_scattered), self.g)
            phi_prim = np.random.uniform(0, 1, n_scattered) * 2 * np.pi
            cos_p = np.cos(phi_prim)
            sin_p = np.sin(phi_prim)
            cos_t = cos_theta_prim
            sin_t = np.sqrt(1 - cos_theta_prim**2)

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
            "photons absorbed by surface": n_hit_ground,
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

        for i, h in history.items():   
            X, Y, Z = np.array(h).T #last_positions[i, :, np.newaxis]))
            ax.plot(X, Y, self.tau_star - Z, label=f'{i}', alpha=0.5)
            ax.scatter(X[-1], Y[-1], self.tau_star - Z[-1])

        fig.savefig(self.img_dir / name)

    def print_results(self):
        if self.results is not None:
            results = self.results
        else:
            raise KeyError("No results found. Use '.run()' first")

        for k, v in results.items():
            print(f"{k}: {v}") if k != "sample paths" else ...


if __name__ == '__main__':
    config = read_config()
    sim = MCRadiation(config)
    sim.run()
    sim.plot_paths()
    sim.print_results()

