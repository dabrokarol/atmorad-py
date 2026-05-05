# atmorad.py
## A vectorized monte carlo simulation of atmospheric radiative transfer. 

## Description:


## How to run:
### Use UV (simplest)

- If not installed, install uv, a fast python package manager: [installation guide](https://docs.astral.sh/uv/getting-started/installation/)
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


## Programmers' assumptions:
- Simulation uses `numpy` to parallelize photons and improve performance
- `Scene` class computes photons' paths and keeps track of the environment
- `Atmosphere` and `Surface` classes keep track of optical properties, phase functions, reflection functions etc.

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

## References and Literature
- (in Polish) Script for Lecture about [Radiative Processes in the Atmosphere](https://www.igf.fuw.edu.pl/~kmark/stacja/wyklady/ProcesyRadiacyjne/2013/WykladRadiacjaKlimat.pdf), K. Markowicz, Faculty of Physics, University of Warsaw, 2013
