from atmorad.constants import PRECOMPUTED_RESOLUTION, EPSILON
from atmorad.physics.registry import register_scattering

import numpy as np

class Scattering:
    def __init__(self, pdf_array, resolution):
        """
        Takes a raw probability density array, normalizes it, 
        and computes the Cumulative Distribution Function (CDF) for fast sampling.
        """
        self.cos_grid = np.linspace(-1, 1, resolution)
        dx = self.cos_grid[1] - self.cos_grid[0]
        pdf_array = pdf_array / (np.sum(pdf_array) * dx)
        self.distribuant = np.cumsum(pdf_array) * dx
        self.n_precomputed = resolution

    def scatter(self, rand_1, rand_2):
        """Computes sin and cos of theta, phi used for scattering. Uses `np.interp` to obtain reversed cdf values for given rand_1. Samples phi from uniform distribution [0,2pi].
        
        Args:
            rand_1 - array of random numbers (uniform(0,1)) used to sample cos_theta
            rand_2 - array of random numbers (uniform(0,1)) used to sample sin_theta

        Returns:
            np.array((cos_theta, sin_theta, cos_phi, sin_phi)) - trigonometric functions of sampled angles
        """
        phi = 2 * np.pi * rand_2
        
        cos_theta = np.interp(rand_1, self.distribuant, self.cos_grid)
        sin_theta = np.sqrt(1 - np.clip(cos_theta**2, 0.0, 1.0))
        
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        
        return np.array((cos_theta, sin_theta, cos_phi, sin_phi))
    
    def __call__(self, rand_1, rand_2):
        return self.scatter(rand_1, rand_2)

@register_scattering("hg")
class HenyeyGreensteinScattering(Scattering):
    def __init__(self, g: float, resolution=PRECOMPUTED_RESOLUTION):
        self.g = g
        cos_grid = np.linspace(-1, 1, resolution)
        
        if np.isclose(g, 1.0, atol=EPSILON):
            pdf = np.isclose(cos_grid, 1.0, atol=EPSILON).astype(float)
        elif np.isclose(g, -1.0, atol=EPSILON):
            pdf = np.isclose(cos_grid, -1.0, atol=EPSILON).astype(float)
        else:
            pdf = (1 - g**2) / (2 * (1 + g**2 - 2 * g * cos_grid)**1.5)
            
        super().__init__(pdf_array=pdf, resolution=resolution)

@register_scattering("isotropic")
class IsotropicScattering(Scattering):
    def __init__(self, resolution=PRECOMPUTED_RESOLUTION):
        cos_grid = np.linspace(-1, 1, resolution)
        pdf = np.ones_like(cos_grid)
        super().__init__(pdf_array=pdf, resolution=resolution)

@register_scattering("rayleigh")
class RayleighScattering(Scattering):
    def __init__(self, resolution=PRECOMPUTED_RESOLUTION):
        cos_grid = np.linspace(-1, 1, resolution)
        pdf = 1.0 + cos_grid**2
        super().__init__(pdf_array=pdf, resolution=resolution)