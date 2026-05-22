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
This project simulates the propagation of light through a plane-parallel atmosphere over a horizontally mixed surface and its interactions with the ground boundary. Developed as a student project to explore computational physics and software engineering, it uses `numpy` to simulate photons simultaneously in large batches via multiprocessing, and generates visualizations using `matplotlib` and `seaborn`.

## Installation

Using [`uv`](https://docs.astral.sh/uv/getting-started/installation/) (Recommended):
```bash
uv tool install atmorad-py
```

Using `pip`:
```bash
pip install atmorad-py
```

## Quickstart

Initialize a default configuration file in your current directory:
```bash
atmorad --init
```

Run the simulation:
```bash
atmorad simulation.toml
```
*Check the `results/` directory for generated simulation artifacts and plots.*

## Physical Model
- **Analog Monte Carlo Approach**: Light is simulated using discrete photon packets. Final flux is calculated as a fraction of the total detected packets.
- **Plane-parallel approximation**: The atmosphere consists of horizontally uniform layers.
- **Multi-material atmospheric layers**: Layers can consist of multiple atmospheric materials simultaneously. A photon is assigned a material randomly when it is initialized and again when it crosses into a new layer. Each material has its own extinction coefficient, SSA, and phase function.
- **Custom Phase Functions**: Henyey-Greenstein and Rayleigh phase functions are built-in, but any custom user-defined function can be constructed using the `Scattering` class.
- **Surface Reflections**: The surface consists of materials, each with its own albedo, a predefined BRDF reflection model (`Lambertian`, `Specular`), and a `ProceduralMap` that outputs material IDs based on spatial coordinates.
- **Photon Properties**: Light is treated as monochromatic, non-polarized particles. During the simulation they can be scattered, reflected, or absorbed. 
- **Incident Irradiance & Adjacency Effect**: Custom detectors allow measuring downward/upward incident flux at any arbitrary altitude, which is helpful for visualizing the adjacency effect.

## Customization
You can define your own surface maps, surface reflection algorithms, and scattering phase functions using decorators as shown below:

```python
import numpy as np
import atmorad
from atmorad import SurfaceReflection, register_reflection, orientation
from atmorad import Scattering, register_scattering
from atmorad import BaseSurfaceMap, register_surface_map
from atmorad.constants import X, Y

# 1. Registering a custom surface map
@register_surface_map("custom-stripe-y", ["material_name_a", "material_name_b"])
class StripeYMap(BaseSurfaceMap):
    # Specify arbitrary custom parameters
    def __init__(self, stripe_width_km: float):
        self.width = stripe_width_km

    def get_material_ids(self, pos: np.ndarray) -> np.ndarray:
        # Maps 2D photon coordinates to integer indices (0 for material_name_a, 1 for material_name_b)
        # Your custom geometry logic here
        grid_x = np.mod(pos[X], self.width)
        return np.where(grid_x < (self.width / 2.0), 0, 1)

# 2. Registering a custom surface reflection
@register_reflection("custom-reflection")
class CustomReflection(SurfaceReflection):
    def __init__(self, param_1, param_2):
        self.param_1 = param_1
        self.param_2 = param_2

    def reflect(self, direction, rand_1, rand_2):
        cos_theta = np.sqrt(rand_1)
        sin_theta = np.sqrt(1.0 - rand_1)
        phi = rand_2 * 2 * np.pi
        cos_phi, sin_phi = np.cos(phi), np.sin(phi)
        return orientation(cos_theta, sin_theta, cos_phi, sin_phi)
    
# 3. Registering a custom scattering phase function
@register_scattering("custom-scattering")
class CustomScattering(Scattering):
    def __init__(self, g, resolution=1000):
        self.assymetry_factor = g
        cos_grid = np.linspace(-1, 1, resolution)
        pdf = (1 - g**2) / (2 * (1 + g**2 - 2 * g * cos_grid) ** 1.5)
        super().__init__(pdf_array=pdf, resolution=resolution)

if __name__ == "__main__":
    # 4. Running the simulation from config which uses your custom names
    results = atmorad.run("simulation.toml")
    # Figures and metadata are saved automatically
    # You can also process results on your own
```

In `simulation.toml` you can specify your defined scatterings, reflections and surface maps:
```toml
[atmosphere_materials.custom-atm-material]
ssa = 0.9
scattering = {type = "custom-scattering", g=0.8} 

[surface_materials.custom-surf-material-a]
albedo = 0.5
reflection = {type = "custom-reflection", param_1=2, param_2=1.3}

[surface_materials.custom-surf-material-b]
albedo = 0.1
reflection = {type = "lambertian"}
```

Then you can use your defined materials for atmospheric layers and the surface:
```toml
[[layer]]
thickness_km = 2
materials = [{type = "custom-atm-material", weight = 1.0}]

# ...

[surface]
name = "custom-stripe-y"
stripe_width_km = 5.0 # include parameters required for this map type
material_name_a = "custom-surf-material-a"
material_name_b = "custom-surf-material-b"
```
## Loading Results
Once a simulation finishes, you can easily load the results and the exact configuration used into a Python environment (like a Jupyter Notebook) for analysis:

```python
import atmorad
import matplotlib.pyplot as plt

# 1. Loading the completed simulation
config, results = atmorad.load("results/demo001")

# 2. Accessing your physical data as NumPy arrays
map_2d = results["surface_flux_map_2d"]

# 3. Analysis or plotting
plt.imshow(map_2d)
plt.title(f"Flux Map for {config.metadata.experiment_name}")
plt.show()
```
## Project Structure
- `engine/`: Divides photons into batches and runs the simulation.
- `physics/`: Contains a rotation function, scattering phase functions, reflection functions.
- `environment/`: Keeps track of the environment. Contains `Scene`, `Atmosphere` and `Surface` classes.
- `detectors/`: Provides functionality for tracking photons during the simulation and generates results.
- `output/`: Handles results and figure generation.
- `config/` and `builder.py`: Parses `.toml` configuration file and generates simulation context.
- `cli.py`: Provides CLI for `atmorad`.

## References and Literature
- (in Polish) Script for Lecture about [Radiative Processes in the Atmosphere](https://www.igf.fuw.edu.pl/~kmark/stacja/wyklady/ProcesyRadiacyjne/2013/WykladRadiacjaKlimat.pdf), Prof. K. Markowicz, Faculty of Physics, University of Warsaw, 2013.

## Acknowledgments
- This project was inspired by the lectures on *Radiative Processes in the Atmosphere* by Prof. K. Markowicz, Faculty of Physics, University of Warsaw.
- Large Language Models were used for code debugging and learning best Python practices (e.g. `dataclasses`, `__init__.py` import interfaces, class responsibilities, config parsing).

## Contributing
Feel free to open an [Issue](https://github.com/dabrokarol/atmorad-py/issues) or submit a Pull Request if you'd like to contribute or report a bug.