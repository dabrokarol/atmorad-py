# AtmoRad
## A vectorized Monte Carlo simulation of atmospheric radiative transfer.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

| **2D Surface absorption map** | **Sample photon paths** |
| :--- | :--- |
| ![map](docs/img/surface_absorption_map.png) | ![paths](docs/img/3d_photon_paths.png) |
| **Vertical flux profile** | **Vertical absorption profile** |
| ![profile](docs/img/vertical_flux_profile.png)| ![hist](docs/img/heating_profile.png) |

## Overview:

This project simulates propagation of light through heterogenous, plane-parallel atmosphere and their interactions with mixed surface boundaries. It is my student project that I created to learn computational physics and software development.

### Physical model
- **Discrete photons**: Photons are treated as discrete particles, not as variable packets of energy. Energy is counted as a fraction of total photons. 
- **Plane-parallel approximation**: Atmosphere consists of horizontally uniform layers.
- **Multi-material atmospheric layers**: layers can consist of a few atmospheric materials simultaneously (photon gets assigned material randomly upon each scattering event). Each material has its own optical density, single-scattering albedo and phase function.
- **Custom Phase-Functions**: Henyey-Greenstein and Rayleigh phase function are already implemented in the simulation, but any custom user-defined function can be constructed using the `Scattering` class.
- **Surface Reflections**: Surface consists of materials, each of which having its albedo, a predefined reflection (`Lambertian`, `Mirror`) and a `ProceduralMap` that outputs material ID based on coordinates.
- **Photon Properties**: Light is treated as monochromatic, non-polarized particles. During the simulation they can get scattered, reflected or absorbed. 


## Technical implementation:
- Simulation uses `numpy` to simulate photons simultaneously in large batches.
- Results are plotted using `matplotlib` and `seaborn` (eg. photon paths, flux profile, 2d ground flux map)
- Code uses multiprocessing to run batches in parallel.
  
## How to run:
### Use UV (Recommended & Fastest)
- If not installed, [install uv](https://docs.astral.sh/uv/getting-started/installation/), a very fast Python package manager.
- Create a virtual environment and install dependencies:
```bash
uv venv
uv pip install -e .
```
- Modify parameters `default_config.toml` or create a custom config.
- Run the simulation:
```bash
uv run main.py <path-to-config.toml>
```
- Check `results/` directory for simulation outputs and plots

## Project Structure
```
.
├── default_config.toml
├── examples/
├── LICENSE
├── main.py
├── pyproject.toml
├── README.md
├── src
│   └── atmorad
│       ├── __init__.py
│       ├── config
│       │   ├── config.py
│       │   └── parser.py
│       ├── detectors
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── atmosphere_heating.py
│       │   ├── paths.py
│       │   ├── plane_flux.py
│       │   └── surface_toa_flux.py
│       ├── engine
│       │   ├── __init__.py
│       │   ├── batch.py
│       │   ├── engine.py
│       │   └── runner.py
│       ├── environment
│       │   ├── __init__.py
│       │   ├── atmosphere.py
│       │   ├── scene.py
│       │   └── surface.py
│       ├── physics
│       │   ├── __init__.py
│       │   ├── geometry.py
│       │   ├── reflection.py
│       │   └── scattering.py
│       ├── data_io.py
│       ├── constants.py
│       └── results.py
```
### Core Architecture:
- `engine/`: divides photons into batches and runs the simulation.
- `Scene`: keeps track of the environment.
- `Atmosphere` and `Surface`: keep track of optical properties, phase functions, reflection functions and layer structures.
- `ResultAnalyzer`: Generates plots based on simulation results.

### Customization:
See `main.py` for examples and comments on how to build custom surface maps and atmospheric layers.

## References and Literature
- (in Polish) Script for Lecture about [Radiative Processes in the Atmosphere](https://www.igf.fuw.edu.pl/~kmark/stacja/wyklady/ProcesyRadiacyjne/2013/WykladRadiacjaKlimat.pdf), Prof. K. Markowicz, Faculty of Physics, University of Warsaw, 2013.

## Contributing:
Feel free to open an Issue or submit a Pull Request if you'd like to contribute or report a bug.

## Acknowledgments
- This project was inspired by the lectures on *Radiative Processes in the Atmosphere* by Prof. K. Markowicz, Faculty of Physics, University of Warsaw.
- Large Language Models were used for code-debugging and learning best python practices (e.g. `dataclasses`, `__init__.py` import interfaces, class responsibilities, config parsing).