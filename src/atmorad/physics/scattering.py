import numpy as np

from atmorad.constants import EPSILON
from atmorad.registry import register_scattering


class Scattering:
    def __init__(self, pdf_array):
        """
        Takes a raw probability density array, normalizes it,
        and computes the Cumulative Distribution Function (CDF) for fast sampling.
        """
        self.n_precomputed = len(pdf_array)
        self.cos_grid = np.linspace(-1, 1, self.n_precomputed)
        dx = self.cos_grid[1] - self.cos_grid[0]
        pdf_array = pdf_array / (np.sum(pdf_array) * dx)
        self.distribuant = np.cumsum(pdf_array) * dx

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
    def __init__(self, g: float):
        self.g = g
        pass

    def scatter(self, rand_1, rand_2):
        if abs(self.g) < EPSILON:
            cos_theta = 2.0 * rand_1 - 1.0
        else:
            sq = (1.0 - self.g**2) / (1.0 - self.g + 2.0 * self.g * rand_1)
            cos_theta = (1.0 + self.g**2 - sq**2) / (2.0 * self.g)

        sin_theta = np.sqrt(1.0 - np.clip(cos_theta**2, 0.0, 1.0))
        phi = 2.0 * np.pi * rand_2
        return np.array((cos_theta, sin_theta, np.cos(phi), np.sin(phi)))

    def __call__(self, rand_1, rand_2):
        return self.scatter(rand_1, rand_2)


@register_scattering("isotropic")
class IsotropicScattering(Scattering):
    def __init__(self):
        pass

    def scatter(self, rand_1, rand_2):
        cos_theta = 2.0 * rand_1 - 1.0
        sin_theta = np.sqrt(1.0 - cos_theta**2)

        phi = 2.0 * np.pi * rand_2
        return np.array((cos_theta, sin_theta, np.cos(phi), np.sin(phi)))

    def __call__(self, rand_1, rand_2):
        return self.scatter(rand_1, rand_2)


@register_scattering("rayleigh")
class RayleighScattering(Scattering):
    def __init__(self):
        pass

    def scatter(self, rand_1, rand_2):
        u = 2.0 * rand_1 - 1.0
        w = np.cbrt(2.0 * u + np.sqrt(4.0 * u**2 + 1.0))
        cos_theta = w - 1.0 / w

        sin_theta = np.sqrt(1.0 - np.clip(cos_theta**2, 0.0, 1.0))
        phi = 2.0 * np.pi * rand_2
        return np.array((cos_theta, sin_theta, np.cos(phi), np.sin(phi)))

    def __call__(self, rand_1, rand_2):
        return self.scatter(rand_1, rand_2)
