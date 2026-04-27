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
    pass
