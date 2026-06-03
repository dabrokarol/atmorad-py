# AtmoRad
## A vectorized Monte Carlo simulation of atmospheric radiative transfer.

[![PyPI version](https://img.shields.io/pypi/v/atmorad-py.svg?color=blue)](https://pypi.org/project/atmorad-py/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Physics: Radiative Transfer](https://img.shields.io/badge/Physics-Radiative_Transfer-ff8c00)](#)
[![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![xarray](https://img.shields.io/badge/xarray-000000?logo=xarray&logoColor=white)](https://xarray.dev/)
[![h5netcdf](https://img.shields.io/badge/h5netcdf-4B8BBE)](https://github.com/h5netcdf/h5netcdf)
[![uv](https://img.shields.io/badge/uv-fast_python_packager-DE5FE9)](https://docs.astral.sh/uv/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://docs.astral.sh/ruff/)
[![pytest](https://img.shields.io/badge/pytest-0A9EDC?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![CI](https://github.com/dabrokarol/atmorad-py/actions/workflows/ci.yml/badge.svg)](https://github.com/dabrokarol/atmorad-py/actions)

| **2D Surface absorption map** | **Sample photon paths** |
| :--- | :--- |
| ![map](https://raw.githubusercontent.com/dabrokarol/atmorad-py/main/docs/img/surface_absorption_map.png) | ![paths](https://raw.githubusercontent.com/dabrokarol/atmorad-py/main/docs/img/3d_photon_paths.png) |
| **Vertical flux profile** | **Vertical absorption profile** |
| ![profile](https://raw.githubusercontent.com/dabrokarol/atmorad-py/main/docs/img/vertical_flux_profile.png)| ![hist](https://raw.githubusercontent.com/dabrokarol/atmorad-py/main/docs/img/absorption_profile.png) |

## Overview
AtmoRad is a Python tool for simulating the radiative transfer of monochromatic light over a mixed 2D surface and a plane-parallel atmosphere. I started it as an independent educational project alongside lectures of Radiative Processes in the Atmosphere at the Faculty of Physics, University of Warsaw to learn computational physics and software development. 

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

Initialize a default configuration file in your current directory:
```bash
> atmorad --init
```

Run the simulation:
```bash
> atmorad simulation.toml
demo001/baseline: 100%|███████████████████████████████| 400000/400000 [00:10<00:00, 37632.74 photons/s]

---- Simulation Summary: demo001/baseline ----
Time: 10.63s (Total) | 35.49s (CPU)
Simulated Photons: 400_000

Energy Distribution:
  Outgoing (TOA)         :  64.35%
  Surface Absorption     :  33.57%
  Atmospheric Absorption :   2.08%
  ------------------------------
  Energy Conservation    : 100.00%

Outputs saved to: results/demo001/
  └─ atmorad_demo001_baseline.nc

```
*Check the `results/` directory for generated simulation artifacts and plots.*

## Features & Physical Model
- **Vectorized Monte Carlo Approach**: For high-performance execution, uses NumPy and multiprocessing for fast and parallel processing of photons in large batches.
- **3D Radiative Transfer in Plane-Parallel Approximation**: The atmosphere consists of horizontally uniform layers, but photon paths are tracked in fully 3D space over a 2D surface.
- **Multi-Material Atmospheric Layers**: Layers can consist of multiple atmospheric materials simultaneously. Each material defines its own extinction coefficient, SSA, and phase function (built-in Rayleigh and Henyey-Greenstein, or custom).
- **Two Scattering Mechanisms**: Supports photon scattering using analytical inverse phase functions as well as numerical inverse CDFs for custom distributions.
- **Surface Reflections**: The surface consists of materials with specific albedos, predefined BRDF reflection models (`lambertian`, `specular`), and a `ProceduralMap` mapping material IDs to spatial coordinates.
- **Photon Properties**: Light is treated as monochromatic, non-polarized, weighted particles that can be scattered, reflected, and partially absorbed.
- **Checkpointing & Data Formats**: Supports resuming from checkpoints in case of interruption. Results are stored in the **NetCDF/HDF5** standard. Results are self-contained in a single `.nc` file.

## Roadmap
I'm planning to include more features in the future, such as:
- Delta tracking for arbitrary 3D cloud geometries.
- Wavelength-dependent optical properties of materials.
- Roughness parameter in specular reflection and other BRDF models.
- 3D surface topography.
- Spherical geometry for high zenith angles and whole-Earth simulations.
- Validation against standard 3D radiative transfer models.

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
[[scenario]]
name    = "checkerboard-surface"
surface = { type = "checkerboard", tile_size_km = 10.0, material_a = "snow", material_b = "ocean" }

[[scenario]]
name               = "less-photons"
engine.num_photons = 50_000
engine.batch_size  = 12_500

# multiple sweep blocks are combined using a cartesian product
[[sweep]]
parameter = "source.zenith_angle_deg"
values    = [0, 15, 30, 45, 60]

[[sweep]]
parameter = "source.azimuth_angle_deg"
values    = [0, 45, 90, 135, 180]
```
<!-- [[[end]]] -->

</details>

### Running Multiple Scenarios:
You can run multiple scenarios by appending [[scenario]] blocks to the end of your TOML file. You can override any base variable using dot-notation.
```toml
# Overrides the solar angle to 30 degrees
[[scenario]]
name = "sun_30"
source.theta_sun_deg = 30

# Overrides both the solar angle and the photon count
[[scenario]]
name = "sun_60"
engine.num_photons = 500_000
source.theta_sun_deg = 60
```

## Loading Results
Because AtmoRad saves data in the standard NetCDF4/HDF5 format, you can read the `data.nc` file directly using libraries like `xarray` or `netCDF4`.

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

# access numbers
total_reflected_energy = ds["energy_toa_outgoing"].item()
total_absorbed_surf = ds["energy_surface_absorbed"].item()

# access attributes
num_photons = ds.attrs["num_photons"]
sim_time = ds.attrs["simulation_time_s"]
active_detectors = ds.attrs["active_detectors"]

```
<!-- [[[end]]] -->

### Extracting configuration file from results
Each data `.nc`  file contains configuration data used to run the simulation. You can extract it by running:
```bash
atmorad --extract-config <path-to-data.nc>
```
This method created an <exp_name>_<scen_name>_config.toml file in your working directory.

## Project Structure
- `engine/`: Handles photon batching and runs the main simulation loop.
- `physics/`: Contains rotation functions, scattering phase functions, and reflection models.
- `environment/`: Manages the environment (`Scene`, `Atmosphere`, `Surface`).
- `detectors/`: Implements photon tracking and result generation.
- `models/`: Defines base classes used throughout the program (and extensive 'results.py' for parsing netcdf).
- `output/`: Handles data IO and figure generation.
- `config/`: Parses `.toml` configuration files and constructs the simulation context.
- `cli.py`: Command-line interface entry point.

## References and Literature
- (in Polish) Script for Lecture about [Radiative Processes in the Atmosphere](https://www.igf.fuw.edu.pl/~kmark/stacja/wyklady/ProcesyRadiacyjne/2013/WykladRadiacjaKlimat.pdf), Prof. K. Markowicz, Faculty of Physics, University of Warsaw, 2013.

## Acknowledgments
- I created this project inspired by the lectures on *Radiative Processes in the Atmosphere* by Prof. K. Markowicz, Faculty of Physics, University of Warsaw.
- I used Large Language Models for code debugging and architectural decisions (e.g., how to structure the repository, which packages to use, how to save and read data).

## Contributing
Feel free to open an [Issue](https://github.com/dabrokarol/atmorad-py/issues) or submit a Pull Request to report bugs, suggest new features and ask questions :))