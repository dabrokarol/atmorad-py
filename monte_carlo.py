import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

#### ASSUMPTIONS (TO EASE LATER)
# Isotropic atmoshperic layer (tau, albedo)
# Perfectly black surface layer (can be changed to brdf)
# Sun position constant (could be taken from some database)
#### OTHER TODOS
# Add consistent colors for plotting

#### DIRECTORIES
img_dir = Path.cwd() / "img"
img_dir.mkdir(exist_ok=True)

#### SIMULATION PARAMETERS
N = 1000
TRACK_N = 20 # Number of photons which paths will be tracked 

#### ATMOSPHERIC PARAMETERS
TAU = 2
OMEGA = 0.99 # single scattering albedo

#### SUNLIGHT ORIENTATION
THETA_SUN = np.pi / 3 # 60 degrees
PHI_SUN = 0 # for simplicity

#### GROUND PARAMETERS
SURFACE_ALBEDO = 0

#### STARTING POSITION
z = 0 # highest point of the atmosphere
x = 0
y = 0

#### PHASE FUNCTION - HENYEY-GREENSTEIN function
G = 0.7

def henyey_greenstein(theta):
    return 0.5 * (1 - G**2 ) / (1 + G**2 - 2*G * np.cos(theta))**(3/2)
def distribuant(theta):
    return (1 - G**2) / (2 * G) * (1 / (1 + G) - 1/np.sqrt(1 + G**2 - 2*G*np.cos(theta)))

if not np.isclose(G, 0):
    def cos_theta(r):
        return 1 / (2*G) * (1 + G**2 - ((1 - G**2) / (2*G*r - G + 1))**2)
else:
    def cos_theta(r):
        return 2 * r - 1

def orientation(theta, phi):
    return np.array((np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)))

def rotate(ori, cos_t, sin_t, cos_p, sin_p):
    result = np.zeros_like(ori)
    big_z = np.abs(ori[2]) > 0.999
    small_z = ~big_z # inverted mask

    sqrt_z = np.sqrt(1 - ori[2, small_z]**2)

    result[:, small_z] = np.array((
        sin_t[small_z] / sqrt_z * (ori[0, small_z] * ori[2, small_z] * cos_p[small_z] - ori[1, small_z] * sin_p[small_z]) + ori[0, small_z] * cos_t[small_z],
        sin_t[small_z] / sqrt_z * (ori[1, small_z] * ori[2, small_z] * cos_p[small_z] + ori[0, small_z] * sin_p[small_z]) + ori[1, small_z] * cos_t[small_z],
        - sin_t[small_z] * cos_p[small_z] * sqrt_z + ori[2, small_z] * cos_t[small_z]
    ))
    result[:, big_z] = np.array((
        sin_t[big_z] * cos_p[big_z], sin_t[big_z] * sin_p[big_z], np.sign(ori[2, big_z]) * cos_t[big_z]
    ))
    
    return result

class MCRadiation:
    def __init__(self):
        self.n_photons = N
        self.n_track = TRACK_N
        
        self.optical_depth = TAU
        self.ss_albedo = OMEGA
        self.sun_theta = THETA_SUN
        self.sun_phi = PHI_SUN

        self.starting_pos = np.array((0, 0, 0))

        self.g = G

        self.results = None

    def _init_arrays(self):
        thetas = np.random.normal(self.sun_theta, 1/60, size=(self.n_photons))
        phis = np.random.normal(self.sun_phi, 1/60, size=(self.n_photons))
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
            hit = (pos[2] > TAU)

            last_pos_left = pos[:, left] + (0 - pos[2, left]) / ori[2, left] * ori[:, left]
            last_pos_hit = pos[:, hit] + (TAU - pos[2, hit]) / ori[2, hit] * ori[:, hit]
            last_positions[ids[left]] = last_pos_left.T
            last_positions[ids[hit]] = last_pos_hit.T

            n_left_atmosphere += np.count_nonzero(left)
            n_hit_ground += np.count_nonzero(hit)

            pos[2, left] = np.inf
            pos[2, hit] = np.inf
            
            w = np.random.uniform(0, 1, n_photons)

            scattered = (w < OMEGA) & (pos[2] < np.inf)
            absorbed = (w >= OMEGA) & (pos[2] < np.inf)

            last_positions[ids[absorbed]] = pos[:, absorbed].T
            pos[2, absorbed] = np.inf

            n_absorbed_atmoshpere += np.count_nonzero(absorbed)
            n_scattered = np.count_nonzero(scattered)
        
            #### HEYNEY GREENSTEIN phase function
            cos_theta_prim = cos_theta(np.random.uniform(0, 1, n_scattered))
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

    def plot_paths(self, name:str='multi.png'):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        if self.results is not None:
            history = self.results['sample paths']
        else:
            raise KeyError("No photon paths found. Use '.run()' first")

        for i, h in history.items():   
            X, Y, Z = np.array(h).T #last_positions[i, :, np.newaxis]))
            ax.plot(X, Y, TAU - Z, label=f'{i}', alpha=0.5)
            ax.scatter(X[-1], Y[-1], TAU - Z[-1])

        fig.savefig(img_dir / name)

    def print_results(self):
        if self.results is not None:
            results = self.results
        else:
            raise KeyError("No results found. Use '.run()' first")

        for k, v in results.items():
            print(f"{k}: {v}") if k != "sample paths" else ...


if __name__ == '__main__':
    sim = MCRadiation()
    sim.run()
    sim.plot_paths()
    sim.print_results()

