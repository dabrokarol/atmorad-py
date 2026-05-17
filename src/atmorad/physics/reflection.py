from abc import ABC, abstractmethod
import logging

import numpy as np

from atmorad.physics.geometry import orientation
from atmorad.constants import X, Y, Z
class SurfaceReflection(ABC):
    @abstractmethod
    def reflect(self, direction: np.ndarray, rand_1: np.ndarray, rand_2: np.ndarray) -> np.ndarray:
        """
        Calculates the new direction of a photon after hitting the surface.
        """
        pass
    
    def __call__(self, direction, rand_1, rand_2):
        return self.reflect(direction, rand_1, rand_2)


class MirrorReflection(SurfaceReflection):
    def __init__(self, roughness: float = 0.0):
        self.roughness = roughness
        
    def reflect(self, direction, rand_1, rand_2):
        new_direction = direction
        new_direction[Z] = np.abs(new_direction[Z])
        
        if self.roughness > 0.0:
            logging.warning("Mirror reflection with roughness > 0 is not yet implemented")

        return new_direction


class LambertianReflection(SurfaceReflection):
    def reflect(self, direction, rand_1, rand_2):
        phi = rand_2 * 2 * np.pi
        
        cos_theta = np.sqrt(rand_1) 
        sin_theta = np.sqrt(1.0 - rand_1)
        
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        return orientation(cos_theta, sin_theta, cos_phi, sin_phi)


class UniformReflection(SurfaceReflection):
    def reflect(self, direction, rand_1, rand_2):
        phi = rand_2 * 2 * np.pi
        
        cos_theta = rand_1
        sin_theta = np.sqrt(1.0 - rand_1**2)
        
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        return orientation(cos_theta, sin_theta, cos_phi, sin_phi)