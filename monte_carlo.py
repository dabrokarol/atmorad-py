import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

#### ASSUMPTIONS (TO EASE LATER)
# Uniform phase function
# Isotropic atmoshperic layer (tau, albedo)
# Perfectly black surface layer (can be changed to brdf)
# Sun position constant (could be taken from some database)

#### DIRECTORIES
img_dir = Path.cwd() / "img"
img_dir.mkdir(exist_ok=True)

#### SIMULATION PARAMETERS
N = 1000
TRACK_N = 10

#### ATMOSPHERIC PARAMETERS
TAU = 10
H = 1000 # [m]
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

#### PHASE FUNCTION - uniform for now

def orientation(theta, phi):
    return np.array((np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), np.sin(theta)))

def rotate(ori, theta, phi):
    result = np.zeros_like(ori)
    big_z = np.abs(ori[2]) > 0.999
    small_z = ~big_z # inverted mask

    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    sin_p = np.sin(phi)
    cos_p = np.cos(phi)
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

def multi_photon(pos, ori, scatter_counts, ids):
    global n_left_atmosphere
    global n_hit_ground
    global n_absorbed_atmoshpere
    global history
    global last_positions

    while ids.size:
        N = ids.size
        T = np.random.uniform(0, 1, N)

        for i, positon in zip(ids[ids < TRACK_N], pos[:, ids < TRACK_N].T):
            history[i].append(positon)

        dist = -np.log(T)

        pos += ori * dist

        left = (pos[2] < 0)
        hit = (pos[2] > TAU)

        last_pos_left = (0 - pos[2, left]) / ori[2, left] * ori[:, left]
        last_pos_hit = (TAU - pos[2, hit]) / ori[2, hit] * ori[:, hit]
        last_positions[ids[left]] = last_pos_left.T
        last_positions[ids[hit]] = last_pos_hit.T

        n_left_atmosphere += np.count_nonzero(left)
        n_hit_ground += np.count_nonzero(hit)

        pos[2, left] = np.inf
        pos[2, hit] = np.inf
        
        w = np.random.uniform(0, 1, N)

        scattered = (w < OMEGA) & (pos[2] < np.inf)
        absorbed = (w >= OMEGA) & (pos[2] < np.inf)

        last_positions[ids[absorbed]] = pos[:, absorbed].T
        pos[2, absorbed] = np.inf

        n_absorbed_atmoshpere += np.count_nonzero(absorbed)
        n_scattered = np.count_nonzero(scattered)

        theta_prim = np.random.uniform(0, 1, n_scattered) * np.pi
        phi_prim = np.random.uniform(0, 1, n_scattered) * 2 * np.pi

        ori[:, scattered] = rotate(ori[:, scattered], theta_prim, phi_prim)
        scatter_counts[scattered] += 1
        
        #### SHRINKING THE ARRAY TO ONLY ALIVE PHOTONS
        msk = pos[2] != np.inf
        pos = pos[:, msk]
        ori = ori[:, msk]
        ids = ids[msk]



thetas_0 = np.random.normal(THETA_SUN, 1, size=(N))
phis_0 = np.random.normal(PHI_SUN, 1, size=(N))
ori_0 = orientation(thetas_0, phis_0)

pos_0 = np.zeros(shape=(3, N), dtype=np.float64)
ids_0 = np.arange(0, N)
scatter_counts_0 = np.zeros(N)

n_left_atmosphere = 0
n_hit_ground = 0
n_absorbed_atmoshpere = 0
history = {i: [] for i in range(TRACK_N)}
last_positions = np.zeros((N, 3))

multi_photon(pos_0, ori_0, scatter_counts_0, ids_0)

print("left atmoshpere:", n_left_atmosphere, "\nhit ground:", n_hit_ground, "\nabsorbed atmosphere:", n_absorbed_atmoshpere)
print("sum:", n_absorbed_atmoshpere + n_hit_ground + n_left_atmosphere)



fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for i, h in history.items():   
    X, Y, Z = np.hstack((np.array(h).T, last_positions[i, :, np.newaxis]))
    ax.plot(X, Y, TAU - Z, label=f'{i}', alpha=0.5)

fig.savefig(img_dir / 'multi.png')