# AtmoRad
## A vectorized Monte Carlo simulation of atmospheric radiative transfer.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

| **2D Surface absorption map** | **Sample photon paths** |
| :--- | :--- |
| ![map](https://raw.githubusercontent.com/dabrokarol/atmorad-py/main/docs/img/surface_absorption_map.png) | ![paths](https://raw.githubusercontent.com/dabrokarol/atmorad-py/main/docs/img/3d_photon_paths.png) |
| **Vertical flux profile** | **Vertical absorption profile** |
| ![profile](https://raw.githubusercontent.com/dabrokarol/atmorad-py/main/docs/img/vertical_flux_profile.png)| ![hist](https://raw.githubusercontent.com/dabrokarol/atmorad-py/main/docs/img/absorption_profile.png) |

## Overview
AtmoRad is a Python framework for 3D atmospheric radiative transfer simulations over a 2D heterogeneous surface and plane-parallel atmosphere using the Monte Carlo method. It leverages `numpy` for efficient batch-photon processing and implements a modular plugin architecture. The project was developed to learn computational physics and software engineering practices.

## Installation

Using [`uv`](https://docs.astral.sh/uv/getting-started/installation/) (Recommended):
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
Simulating Photons: 100%|█████████████████████████████| 100000/100000 [00:04<00:00, 20454.69 photons/s]

---- Simulation Summary: demo001 ----
Time: 4.89s (Total) | 4.86s (CPU)
Total Photons / Energy: 100_000

Energy Distribution:
  Reflected (TOA)      :  63.40%
  Surface Absorbed     :  35.33%
  Atmosphere Absorbed  :   1.26%
  ------------------------------
  Energy Balance       : 100.00%

Outputs saved to: results/demo001/
  ├─ metadata.json
  ├─ data.nc
  └─ runtime_config.toml

```
*Check the `results/` directory for generated simulation artifacts and plots.*

## Features & Physical Model
- **Weight-Based Monte Carlo Approach**: Each photon packet carries a weight representing energy. Upon interaction with the atmosphere or surface, the weight is incrementally reduced according to the Single Scattering Albedo (SSA) or surface albedo.
- **3D Radiative Transfer in Plane-Parallel Approximation**: The atmosphere consists of horizontally uniform layers, but photon paths are tracked in fully 3D space over a 2D surface.
- **Multi-Material Atmospheric Layers**: Layers can consist of multiple atmospheric materials simultaneously. Each material defines its own extinction coefficient, SSA, and phase function (built-in Rayleigh and Henyey-Greenstein, or custom).
- **Surface Reflections**: The surface consists of materials with specific albedos, predefined BRDF reflection models (`lambertian`, `specular`), and a `ProceduralMap` mapping material IDs to spatial coordinates.
- **Photon Properties**: Light is treated as monochromatic, non-polarized particles that can be scattered, reflected, or absorbed.
- **Resilience & I/O**: Supports resuming from checkpoints in case of interruption. Results are stored in the NetCDF/HDF5 standard for multidimensional array data.

## Configuration

<details>
<summary>The simulation is controlled via a TOML configuration file (click to expand).</summary>

```toml
[metadata]
experiment_name = "demo001"
description = "A demo simulation of 3D radiative transfer over a heterogeneous surface."

[engine]
num_photons = 100_000
batch_size = 100_000         # Memory vs speed tradeoff (controls NumPy array sizes)
random_seed = 42             # Set for reproducible stochastic results
cpu_cores = 1
resume_from_checkpoint = false

# Continuous Absorption & Variance Reduction parameters
photon_weight_threshold = 1e-4  # Weight below which Russian Roulette is triggered
photon_survival_chance = 0.1    # 10% chance to survive with 10x multiplied weight

[source]
theta_sun_deg = 30           # Solar zenith angle (0 = directly overhead, 90 = horizon)
phi_sun_deg = 0              # Solar azimuth angle
wavelength_nm = 530          # Monochromatic wavelength reference

[geometry]
domain_size_x_km = 50
domain_size_y_km = 100
boundary_condition = "periodic" # Photons exiting sides will wrap around to the other side

[detectors]
active = [
    "fate", 
    "path_tracking", 
    "vertical_flux", 
    "absorption_vertical", 
    "plane_flux", 
    "surface_absorption"
]

# Spatial resolution for binning energy in detectors
vertical_profiles_resolution_km = 0.5
horizontal_maps_resolution_km = 1.0
num_full_paths = 100
flux_maps_z_levels_km = [0.0, 4.0, 10.0]

[output]
save_plots = true
overwrite = true
path = 'results'

# --- Physical Properties of Materials ---

[surface_materials.snow]
albedo = 0.85                # Reflects 85% of energy, absorbs 15%
reflection = {type = "lambertian"} # Diffuse, perfectly isotropic reflection
[surface_materials.ocean]
albedo = 0.01                # Highly absorptive
reflection = {type = "specular", roughness = 0.0} # Mirror-like reflection

[atmosphere_materials.air]
extinction_coeff_per_km = 0.01 # Clear air, very long mean free path
ssa = 0.9                    # Single Scattering Albedo (Fraction of energy surviving collision)
scattering = {type = "rayleigh"} 
[atmosphere_materials.light_clouds]
extinction_coeff_per_km = 1  
ssa = 0.999999               # Almost no absorption, pure scattering
scattering = {type = "hg", asymmetry_factor = 0.85} # g > 0 means strong forward scattering
[atmosphere_materials.dark_clouds]
extinction_coeff_per_km = 5  # Very dense medium (short mean free path)
ssa = 0.999999
scattering = {type = "hg", asymmetry_factor = 0.85}

# --- 1D Vertical Structure (Bottom to Top) ---

[[layer]]
thickness_km = 2
materials = [{type = "air", weight = 1.0}]

[[layer]]
thickness_km = 4
# Mixtures are supported. Path is calculated for homogenized medium, 
# while collision material is stochastically sampled based on weights.
materials = [
    {type = "air", weight = 0.1},
    {type = "dark_clouds", weight = 0.9}
]

[[layer]]
thickness_km = 4
materials = [{type = "air", weight = 1.0}]

# --- 2D Surface Heterogeneity ---

[surface]
name = "circle"              # Procedural map type registered in the engine
radius_km = 20
material_in = "snow"         # Inside the circle
material_out = "ocean"       # Outside the circle
```
</details>

## Customization (Registry Pattern)
<details>
<summary>
AtmoRad uses a registry pattern, allowing users to define custom surface maps, reflection algorithms, scattering phase functions, and detectors using decorators.</summary>

### Custom Materials and Geometries
```python
import numpy as np
import atmorad
from atmorad import SurfaceReflection, register_reflection, orientation
from atmorad import Scattering, register_scattering
from atmorad import BaseSurfaceMap, register_surface_map
from atmorad.constants import X, Y

# 1. Register a custom surface map
@register_surface_map("custom-stripe-y", ["material_name_a", "material_name_b"])
class StripeYMap(BaseSurfaceMap):
    def __init__(self, stripe_width_km: float):
        self.width = stripe_width_km

    def get_material_ids(self, pos: np.ndarray) -> np.ndarray:
        grid_x = np.mod(pos[X], self.width)
        return np.where(grid_x < (self.width / 2.0), 0, 1)

# 2. Register a custom surface reflection
@register_reflection("custom-reflection")
class CustomReflection(SurfaceReflection):
    def __init__(self, param_1, param_2):
        self.param_1 = param_1
        self.param_2 = param_2

    def reflect(self, direction, rand_1, rand_2):
        # Cosine-weighted hemispherical sampling (e.g., for diffuse reflection)
        cos_theta = np.sqrt(rand_1)
        sin_theta = np.sqrt(1.0 - rand_1)
        
        # Uniform sampling for the azimuth angle
        phi = rand_2 * 2 * np.pi
        cos_phi, sin_phi = np.cos(phi), np.sin(phi)
        
        return orientation(cos_theta, sin_theta, cos_phi, sin_phi)
    
# 3. Register a custom scattering phase function
@register_scattering("custom-scattering")
class CustomScattering(Scattering):
    def __init__(self, g, resolution=1000):
        self.asymmetry_factor = g
        cos_grid = np.linspace(-1, 1, resolution)
        
        # Calculate the Probability Density Function (PDF) 
        # using the Henyey-Greenstein analytical formula
        pdf = (1 - g**2) / (2 * (1 + g**2 - 2 * g * cos_grid) ** 1.5)
        
        # The base class automatically builds the CDF and handles inversion
        super().__init__(pdf_array=pdf, resolution=resolution)

if __name__ == "__main__":
    results = atmorad.run("simulation.toml")
```

To use these in `simulation.toml`:
```toml
[atmosphere_materials.custom-atm-material]
ssa = 0.9
scattering = {type = "custom-scattering", g = 0.8} 

[surface_materials.custom-surf-material-a]
albedo = 0.5
reflection = {type = "custom-reflection", param_1 = 2, param_2 = 1.3}

[surface]
name = "custom-stripe-y"
stripe_width_km = 5.0
material_name_a = "custom-surf-material-a"
material_name_b = "ocean"
```

### Custom Detectors
You can track specific photon behaviors by subclassing `BaseDetector` and implementing its lifecycle hooks.

```python
from dataclasses import dataclass
import numpy as np
from atmorad import BaseDetector, register_detector, SimConfig, Scene, PhotonBatch
@dataclass(slots=True)
class FateResult:
    energy_absorbed_surface: float = 0.0
    energy_absorbed_atmosphere: float = 0.0
    energy_escaped_toa: float = 0.0
    cpu_time_s: float = 0.0

    def merge(self, other: Self):
        return FateResult(
            energy_absorbed_surface=self.energy_absorbed_surface + other.energy_absorbed_surface,
            energy_absorbed_atmosphere=self.energy_absorbed_atmosphere
            + other.energy_absorbed_atmosphere,
            energy_escaped_toa=self.energy_escaped_toa + other.energy_escaped_toa,
            cpu_time_s=self.cpu_time_s + other.cpu_time_s,
        )

@register_detector("fate", FateResult)
class FateDetector(BaseDetector):
    def __init__(self, scene: Scene, config: SimConfig):
        self.absorbed_surface = 0.0
        self.absorbed_atmosphere = 0.0
        self.escaped_toa = 0.0
        self.scene = scene

    def record_movement(self, batch: PhotonBatch, old_pos: np.ndarray):
        pass

    def record_interaction(self, batch, old_direction old_weight, scatter_mask, surface_mask):
        # Continuous Absorption: We calculate deposited energy by subtracting 
        # the photon's new weight from its old weight, rather than counting particles.
        
        if np.any(scatter_mask):
            deposited = old_weight[scatter_mask] - batch.weight[scatter_mask]
            self.absorbed_atmosphere += np.sum(deposited)

        if np.any(surface_mask):
            deposited = old_weight[surface_mask] - batch.weight[surface_mask]
            self.absorbed_surface += np.sum(deposited)
    
    def record_termination(self, batch, terminated_mask):
        if not np.any(terminated_mask):
            return

        term_pos = batch.pos[:, terminated_mask]
        term_weight = batch.weight[terminated_mask]

        escaped_toa_mask = self.scene.above_toa(term_pos)
        if np.any(escaped_toa_mask):
            self.escaped_toa += np.sum(term_weight[escaped_toa_mask])

    def finalize(self):
        pass

    def get_results(self) -> FateResult:
        return FateResult(
            energy_absorbed_surface=self.absorbed_surface,
            energy_absorbed_atmosphere=self.absorbed_atmosphere,
            energy_escaped_toa=self.escaped_toa,
        )
```
</details>

## Loading Results
Simulation results and configurations can be loaded into a Python environment (e.g., Jupyter Notebook) for further analysis in two ways:

### 1. Using the built-in `atmorad.load()`
This method loads both the exact configuration used (as a `SimConfig` instance) and the structured results containing native NumPy arrays.

```python
import atmorad
import matplotlib.pyplot as plt

# Load the completed simulation
config, results = atmorad.load("results/demo001")

# Access physical data as NumPy arrays
map_2d = results.detector_results["surface_absorption"].surface_absorption_map_2d

# Analyze or plot
plt.imshow(map_2d)
plt.title(f"Flux Map for {config.metadata.experiment_name}")
plt.show()
```

### 2. Using standard NetCDF libraries
Because AtmoRad saves data in the standard NetCDF4/HDF5 format, you can read the `data.nc` file directly using widely available scientific libraries such as `xarray` or `netCDF4`.

```python
import xarray as xr

# Open the NetCDF file directly
ds = xr.open_dataset("results/demo001/data.nc", engine="h5netcdf")

# Access variables and attributes ({detector_name}_{attribute_name})
map_2d = ds["surface_absorption_surface_absorption_map_2d"].values
total_escaped_energy = ds.attrs["fate_energy_reflected_toa"]

```
<!-- [[[end]]] -->

## Project Structure
- `engine/`: Handles photon batching and executes the main simulation loop.
- `physics/`: Contains rotation functions, scattering phase functions, and reflection models.
- `environment/`: Manages the environment state (`Scene`, `Atmosphere`, `Surface`).
- `detectors/`: Implements photon tracking and result generation.
- `output/`: Handles NetCDF exports and figure generation.
- `config/`: Parses `.toml` configuration files and constructs the simulation context.
- `cli.py`: Command-line interface entry point.

## References and Literature
- (in Polish) Script for Lecture about [Radiative Processes in the Atmosphere](https://www.igf.fuw.edu.pl/~kmark/stacja/wyklady/ProcesyRadiacyjne/2013/WykladRadiacjaKlimat.pdf), Prof. K. Markowicz, Faculty of Physics, University of Warsaw, 2013.

## Acknowledgments
- This project was inspired by the lectures on *Radiative Processes in the Atmosphere* by Prof. K. Markowicz, Faculty of Physics, University of Warsaw.
- Large Language Models were used for code debugging and architectural decisions (e.g., configuration parsing, public API design).

## Contributing
Feel free to open an [Issue](https://github.com/dabrokarol/atmorad-py/issues) or submit a Pull Request to report bugs or suggest enhancements.