# atmorad.py
## A vectorized monte carlo simulation of atmospheric radiative transfer. 

## How to run:
### Use UV (simplest)

- If not installed, [install uv](https://docs.astral.sh/uv/getting-started/installation/), a fast python package manager
- Create a virtual environment:
```
uv venv
uv pip install -r requirements.txt
```
- Modify `config.toml` and `main.py` to your liking
- Run the simulation 
```bash
uv run main.py
```
- Check `fig/` directory for simulation outputs and plots

<!-- ### The traditional way

- Create a virtual environment:
  ```
  python3 -m venv .venv
  ```
- Modify `config.toml` and `main.py` to your liking
- Activate the environment and run the simulation 
  - Windows
  ```sh
  .venv/scripts/activate
  python3 main.py
  ```
  - Linux / MacOS
  ```sh
  source .venv/bin/activate
  python3 main.py
  ```
- Check `fig/` directory for simulation outputs and plots -->

## Physical assumptions:
Simulation treats photons as monochromatic, non-polarized particles. Environment is created with plane-parallel approximation and can consist of uniform layers of chosen height and material. Each material can have its own scattering phase function, optical density and albedo. Surface consists of a list of materials and a procedural map that outputs material type based on position. 

- Simulated photons are Monochromatic and non-polarized
- Atmospheric layer can consist of a few materials at once so that each photon gets that material assigned randomly upon entering such layer (useful for partially cloudy layers)
- Henyey-Greenstein scattering function is used


## Programmers' assumptions:
- Simulation uses `numpy` to simulate `N` photons in parallel
- Plots are created using `matplotlib`:
  - whole paths for a selected batch of photons (`n_track` in `config.toml`) 
  - aggregated final positions for all photons

## Project Structure
```
.
└── src
    ├── __init__.py
    ├── physics
    │   ├── __init__.py
    │   ├── geometry.py
    │   ├── reflection.py
    │   └── scattering.py
    ├── constants.py
    ├── simulation.py
    ├── scene.py
    ├── atmosphere.py
    └── surface.py
├── config.toml
├── img
│   └── default.png
├── main.py
├── README.md
├── requirements.txt
```

- `Scene` class computes photons' paths and keeps track of the environment
- `Atmosphere` and `Surface` classes keep track of optical properties, phase functions, reflection functions etc.

## Custom Surface and Atmospheric Layers:
See `main.py` for examples and comments.

## References and Literature
- (in Polish) Script for Lecture about [Radiative Processes in the Atmosphere](https://www.igf.fuw.edu.pl/~kmark/stacja/wyklady/ProcesyRadiacyjne/2013/WykladRadiacjaKlimat.pdf), K. Markowicz, Faculty of Physics, University of Warsaw, 2013
