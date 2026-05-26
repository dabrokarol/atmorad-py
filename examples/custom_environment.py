import numpy as np

import atmorad
from atmorad import (
    Scattering,
    register_reflection,
    register_scattering,
    register_surface_map,
)
from atmorad.constants import X
from atmorad.physics import orientation


# 1. Register a custom surface map 
@register_surface_map("custom-stripe-y", ["material_name_a", "material_name_b"])
def stripe_y_map(pos: np.ndarray, stripe_width_km: float) -> np.ndarray:
    """Returns 0 for material A, 1 for material B."""
    grid_x = np.mod(pos[X], stripe_width_km)
    return np.where(grid_x < (stripe_width_km / 2.0), 0, 1)


# 2. Register a custom surface reflection
@register_reflection("custom-reflection")
def custom_reflection(direction: np.ndarray, rand_1: np.ndarray, rand_2: np.ndarray, 
                      param_1: float, param_2: float) -> np.ndarray:
    """
    Cosine-weighted hemispherical sampling.
    Note: param_1 and param_2 are injected directly from TOML.
    """
    cos_theta = np.sqrt(rand_1)
    sin_theta = np.sqrt(1.0 - rand_1)

    phi = rand_2 * 2 * np.pi
    
    return orientation(cos_theta, sin_theta, np.cos(phi), np.sin(phi))


# 3.a. Register a custom numerical scattering phase function
@register_scattering("custom-scattering")
class CustomScattering(Scattering):
    def __init__(self, g: float):
        cos_grid = np.linspace(-1, 1, 1000)

        # Calculate the probability density function
        pdf = (1 - g**2) / (2 * (1 + g**2 - 2 * g * cos_grid) ** 1.5)

        # Calling base class automatically normalizes and builds the numerical inverse
        super().__init__(pdf_array=pdf)


# 3.b. Register a custom analytical scattering phase function (usually better performance)
@register_scattering("custom-scattering-b")
def custom_scattering(rand_1, rand_2, g: float):
    cos_theta = 2.0 * rand_1 - 1.0
    sin_theta = np.sqrt(1.0 - cos_theta**2)

    phi = 2.0 * np.pi * rand_2
    return np.array((cos_theta, sin_theta, np.cos(phi), np.sin(phi)))


if __name__ == "__main__":
    # 4. Run the experiment
    results = atmorad.run("simulation.toml")