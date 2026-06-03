# atmorad
## Monte Carlo atmospheric radiative transfer in Python.

[![PyPI version](https://img.shields.io/pypi/v/atmorad-py.svg?color=blue)](https://pypi.org/project/atmorad-py/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/dabrokarol/atmorad-py/actions/workflows/ci.yml/badge.svg)](https://github.com/dabrokarol/atmorad-py/actions)

| **2D surface absorption map** | **Sample photon paths** |
| :--- | :--- |
| ![map](https://raw.githubusercontent.com/dabrokarol/atmorad-py/main/docs/img/surface_absorption_map.png) | ![paths](https://raw.githubusercontent.com/dabrokarol/atmorad-py/main/docs/img/3d_photon_paths.png) |
| **Vertical flux profile** | **Vertical absorption profile** |
| ![profile](https://raw.githubusercontent.com/dabrokarol/atmorad-py/main/docs/img/vertical_flux_profile.png)| ![hist](https://raw.githubusercontent.com/dabrokarol/atmorad-py/main/docs/img/absorption_profile.png) |
Example outputs generated from the default configuration |(approx 10 seconds for 400_000 photons on thinkpad T480s)

## Overview

atmorad is a Monte Carlo radiative transfer model written in Python.

The code simulates photon transport through a plane-parallel atmosphere above a heterogeneous surface and records quantities such as radiative fluxes, absorption profiles, and surface energy deposition.


### Motivation
I started it as a learning project during lectures on Radiative Processes in the Atmosphere at the Faculty of Physics, University of Warsaw.

The original goal was to better understand Monte Carlo radiative transfer by implementing the underlying algorithms from scratch.

As I wrote more code, it also became an opportunity to learn how to organize a larger codebase.

### Capabilities
- vectorized photon transport using NumPy arrays
- plane-parallel layered atmosphere (photons tracked in 3D)
- heterogeneous 2D surface maps
- Rayleigh and Henyey-Greenstein phase functions
- Lambertian and specular BRDFs
- xarray-compatible NetCDF/HDF5 output
- checkpointing and simulation resume

### Limitations
- monochromatic radiation
- no polarization
- plane-parallel atmosphere
- horizontally homogeneous atmospheric layers
- no validation against reference radiative transfer benchmarks yet
- atmospheric optical properties are currently wavelength-independent

### Future work
- validation against standard 3D radiative transfer models
- delta tracking for arbitrary 3D cloud geometries
- wavelength-dependent optical properties of materials
- roughness parameter in specular reflection and other BRDF models
- 3D surface topography
- spherical geometry for high zenith angles and whole-Earth simulations


## Installation

Using [`uv`](https://docs.astral.sh/uv/getting-started/installation/) (Recommended for project isolation):
```bash
> uv tool install atmorad-py
```

Using `pip`:
```bash
> pip install atmorad-py
```

## Quickstart

### CLI

Initialize a default configuration file in your current directory:
```bash
> atmorad --init
```

Run the simulation:
```bash
> atmorad simulation.toml
demo001/baseline: 100%|█████████████████████████████████████████████████████████| 400000/400000 [00:08<00:00, 48699.42 photons/s]

Simulation complete

experiment: demo001
scenario: baseline

runtime: 8.22 s
photons: 400_000

Energy distribution
------------------------------
toa escape               65.75%
surface absorption       34.23%
atmospheric absorption    0.03%
------------------------------
energy conservation     100.00%


Result File:
  results/demo001/atmorad_demo001_baseline.nc
```
Check the `results/` and `plots/` directories for generated simulation artifacts and plots.

### Python script
```python
from atmorad import run

ds = run("simulation.toml")
```

## Configuration

<details>
<summary>The simulation is controlled via a TOML configuration file (click to expand).</summary>
<!-- [[[cog
import cog
cog.out(f'\n```toml\n{open("src/atmorad/config/simulation.toml").read()}\n```')
]]] -->

```toml
# =============================================================================
# atmorad simulation configuration file
# =============================================================================

[metadata]
experiment_name = "demo001"
description     = "a demo simulation of radiative transfer over a heterogenous surface."

[engine]
random_seed                   = 42
num_photons                   = 400_000
batch_size                    = 100_000  # photons per simulation chunk
roulette_weight_threshold     = 1e-4
roulette_survival_probability = 0.1      # 10% chance to survive with 10x multiplied weight
num_threads                   = 4
resume_from_checkpoint        = false

[source]
zenith_angle_deg  = 30
azimuth_angle_deg = 0
wavelength_nm     = 530  # only for reference, wavelength-dependent parameters not implemented yet

[domain]
size_x_km          = 100
size_y_km          = 100
boundary_condition = "periodic"

# -----------------------------------------------------------------------------
# active detectors (comment out a block to deactivate)
# -----------------------------------------------------------------------------

[detectors.energy_budget]

[detectors.trajectories]
max_tracked_paths = 200

[detectors.flux_profile]
vertical_resolution_km = 0.1

[detectors.absorption_profile]
vertical_resolution_km = 0.2

[detectors.flux_maps]
horizontal_resolution_km = 2.1
z_levels_km              = [0.0, 4.0, 10.0]

[detectors.surface_absorption_map]
horizontal_resolution_km = 2.1

[output]
save_plots = true
overwrite  = true
base_dir   = "results"
fig_dir    = "plots"

# -----------------------------------------------------------------------------
# material types and properties
# -----------------------------------------------------------------------------

[surface_materials.snow]
albedo = 0.85
brdf   = { type = "lambertian" }

[surface_materials.ocean]
albedo = 0.01
brdf   = { type = "specular", roughness = 0.0 }

[atmosphere_materials.air]
extinction_coeff_per_km = 0.01  # optical density
ssa                     = 0.999
phase_function          = { type = "rayleigh" }

[atmosphere_materials.light_clouds]
extinction_coeff_per_km = 1.0
ssa                     = 0.999999  # negligible absorption
phase_function          = { type = "hg", g = 0.85 }  # g > 0 means forward scattering

[atmosphere_materials.dark_clouds]
extinction_coeff_per_km = 5.0
ssa                     = 0.999999
phase_function          = { type = "hg", g = 0.85 }

# -----------------------------------------------------------------------------
# atmospheric layers (bottom to top)
# -----------------------------------------------------------------------------

[[layer]]
thickness_km = 2.0
components   = { air = 1.0 }

[[layer]]
thickness_km = 4.0
components   = { air = 1.0, dark_clouds = 0.9 }  # { material: extinction coefficient multiplier }

[[layer]]
thickness_km = 4.0
components   = { air = 1.0 }

# -----------------------------------------------------------------------------
# surface map configuration (choose one by uncommenting)
# -----------------------------------------------------------------------------

# [surface]
# type     = "uniform"
# material = "snow"

[surface]
type         = "circle"
radius_km    = 20.0
material_in  = "snow"
material_out = "ocean"

# [surface]
# type           = "split_half_x"
# material_left  = "snow"
# material_right = "ocean"

# [surface]
# type         = "checkerboard"
# tile_size_km = 10.0
# material_a   = "snow"
# material_b   = "ocean"

# -----------------------------------------------------------------------------
# scenarios and sweeps (batch experiments)
# -----------------------------------------------------------------------------

# scenario overrides: each block runs a separate simulation
# [[scenario]]
# name    = "checkerboard-surface"
# surface = { type = "checkerboard", tile_size_km = 10.0, material_a = "snow", material_b = "ocean" }

# [[scenario]]
# name               = "less-photons"
# engine.num_photons = 50_000
# engine.batch_size  = 12_500

# multiple sweep blocks are combined using a cartesian product
# [[sweep]]
# parameter = "source.zenith_angle_deg"
# values    = [0, 15, 30, 45, 60]

# [[sweep]]
# parameter = "source.azimuth_angle_deg"
# values    = [0, 45, 90, 135, 180]
```
<!-- [[[end]]] -->

</details>

## Loading results
Results are stored as NetCDF4/HDF5 files and can be loaded directly with xarray:

<!-- [[[cog
import cog
cog.out(f'\n```python\n{open("examples/load_netcdf.py").read()}\n```')
]]] -->

```python
import xarray as xr

# open NetCDF file
ds = xr.open_dataset("results/demo001/atmorad_demo001_baseline.nc", engine="h5netcdf")

# access arrays
map_2d = ds["surface_absorption_map"].values
flux_profile = ds["flux_down_profile"].values

# access scalars
total_reflected_energy = ds["energy_toa_outgoing"].item()
total_absorbed_surf = ds["energy_surface_absorbed"].item()

# access attributes
num_photons = ds.attrs["num_photons"]
sim_time = ds.attrs["simulation_time_s"]
active_detectors = ds.attrs["active_detectors"]

```
<!-- [[[end]]] -->
Or via atmorad:
```python
from atmorad import load
ds = load("results/demo001/atmorad_demo001_baseline.nc")
```

### Extracting configuration file from results
Each data `.nc`  file contains configuration data used to run the simulation. You can extract it by running:
```bash
atmorad --extract-config <path-to-data.nc>
```
This creates an `<exp_name>_<scen_name>_config.toml` file in the current working directory.

## References and literature
- (in Polish) Script for lecture about [Radiative Processes in the Atmosphere](https://www.igf.fuw.edu.pl/~kmark/stacja/wyklady/ProcesyRadiacyjne/2013/WykladRadiacjaKlimat.pdf), Prof. K. Markowicz, Faculty of Physics, University of Warsaw, 2013.

## Acknowledgments
- I created this project inspired by the lectures on *Radiative Processes in the Atmosphere* by Prof. K. Markowicz, Faculty of Physics, University of Warsaw.
- I used Large Language Models as a programming aid during development.

## Contributing
Contributions of any size are welcome. 
- [Open an issue](https://github.com/dabrokarol/atmorad-py/issues) to report a bug or to suggest something.
- You can contact me at atmorad@kdabr.com for any further questions.