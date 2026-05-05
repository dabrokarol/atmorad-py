import numpy as np

class Scattering:
    def __init__(self, scatter_func, g, n_precomputed=1000):
        """Computes normalized probability distribution of cos_theta and sums to obtain distribuant."""
        cos_grid = np.linspace(-1, 1, n_precomputed)
        dx = cos_grid[1] - cos_grid[0]

        pdf = scatter_func(g, cos_grid)
        pdf /= (np.sum(pdf) * dx)

        self.distribuant = np.cumsum(pdf) * dx
        self.cos_grid = cos_grid
        self.n_precomputed = n_precomputed

    def scatter(self, rand_1, rand_2):
        """Computes sin and cos of theta, phi used for scattering. Uses `np.interp` to obtain reversed distribuant values for given rand_1. Samples phi from uniform distribution [0,2pi].
        
        Args:
            rand_1 - array of random numbers (uniform(0,1)) used to sample cos_theta
            rand_2 - array of random numbers (uniform(0,1)) used to sample sin_theta

        Returns:
            np.array((cos_t, sin_t, cos_p, sin_p)) - trigonometric functions of sampled angles
        """
        phi = 2*np.pi*rand_2
        cos_t = np.interp(rand_1, self.distribuant, self.cos_grid)
        sin_t = np.sqrt(1 - np.clip(cos_t**2, 0, 1))
        cos_p = np.cos(phi)
        sin_p = np.sin(phi)
        return np.array((cos_t, sin_t, cos_p, sin_p))
    
    def __call__(self, rand_1, rand_2):
        return self.scatter(rand_1, rand_2)
    
    @staticmethod
    def henyey_greenstein_func(g, cos_t):
        """Henyey-Greenstein function."""
        if np.isclose(g, 1):
            return (np.isclose(cos_t, 1)).astype(float)
        elif np.isclose(g, -1):
            return (np.isclose(cos_t, -1)).astype(float)
        else:
            return (1 - g**2) / (2) / (1 + g**2 - 2*g*cos_t)**(3/2)
    @staticmethod    
    def uniform_func(g, cos_t):
        """Uniform scattering function."""
        return np.ones_like(cos_t, dtype=np.float64)

class HenyeyGreenstein(Scattering):
    def __init__(self, g, n_precomputed=1000):
        super().__init__(self.henyey_greenstein_func, g, n_precomputed)

class Uniform(Scattering):
    def __init__(self, g, n_precomputed=1000):
        super().__init__(self.henyey_greenstein_func, g, n_precomputed)
