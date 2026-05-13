import numpy as np

from atmorad.physics.geometry import orientation

class SurfaceReflection:
    def __init__(self, reflection_func):
        self.reflection_func = reflection_func

    def reflect(self, direction, rand_1, rand_2):
        return self.reflection_func(direction, rand_1, rand_2)
    
    def __call__(self, direction, rand_1, rand_2):
        return self.reflection_func(direction, rand_1, rand_2)
    
    @staticmethod
    def mirror_reflection_func(direction, rand_1, rand_2):
        direction[2] = -direction[2]
        return direction
    
    @staticmethod
    def lambertian_reflection_func(direction, rand_1, rand_2):
        phi = rand_2 * 2 * np.pi

        cos_theta = -np.sqrt(rand_1) # cosine-weighted hemisphere sampling, minus because surface has the biggest height
        sin_theta = np.sqrt(1 - rand_1)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        return orientation(cos_theta, sin_theta, cos_phi, sin_phi)
    
    @staticmethod
    def uniform_reflection_func(direction, rand_1, rand_2):
        theta = rand_1 * np.pi / 2
        phi = rand_2 * 2 * np.pi

        cos_theta = -np.cos(theta) # uniform sampling
        sin_theta = np.sin(theta)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        return orientation(cos_theta, sin_theta, cos_phi, sin_phi)
    
class MirrorReflection(SurfaceReflection):
    def __init__(self):
        super().__init__(self.mirror_reflection_func)

class LambertianReflection(SurfaceReflection):
    def __init__(self):
        super().__init__(self.lambertian_reflection_func)

class UniformReflection(SurfaceReflection):
    def __init__(self):
        super().__init__(self.uniform_reflection_func)