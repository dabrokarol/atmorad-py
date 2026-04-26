import tomllib
import numpy as np
from src.simulation import MCRadiation
from src.atmosphere import Atmosphere
from src.boundaries import Surface, Space

def read_config(path = 'config.toml'):
    try:
        with open(path, 'rb') as f:
            data = tomllib.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"config file missing at {path}")
    return data

if __name__ == '__main__':
    config = read_config()
    atm = Atmosphere(config['atmoshpere'])
    sur = Surface(config['surface'])
    spa = Space()
    sim = MCRadiation(config['simulation'], atm, sur, spa)
    sim.run()
    sim.plot_paths()
    sim.print_results()

