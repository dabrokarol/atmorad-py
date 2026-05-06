from dataclasses import dataclass

@dataclass
class SimConfig:
    num_photons: int = 100_000
    num_track: int = 100
    starting_pos: tuple = (0, 0, 0)
    random_seed: int = 42
    theta_sun_deg: float = 60
    phi_sun_deg: float = 0
    execution_time_s: float|None = None