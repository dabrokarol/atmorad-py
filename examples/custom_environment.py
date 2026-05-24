import numpy as np

import atmorad
from atmorad import (
    BaseSurfaceMap,
    Scattering,
    SurfaceReflection,
    register_reflection,
    register_scattering,
    register_surface_map,
)
from atmorad.constants import X
from atmorad.physics import orientation


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

        # The base class automatically builds the CDF inverse
        super().__init__(pdf_array=pdf, resolution=resolution)


if __name__ == "__main__":
    results = atmorad.run("simulation.toml")
