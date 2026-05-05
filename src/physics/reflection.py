import numpy as np

from src.physics.geometry import orientation

class SurfaceReflection:
    def __init__(self, reflection_func):
        self.reflection_func = reflection_func

    def reflect(self, ori, rand_1, rand_2):
        return self.reflection_func(ori, rand_1, rand_2)
    
    def __call__(self, ori, rand_1, rand_2):
        return self.reflection_func(ori, rand_1, rand_2)
    
    @staticmethod
    def mirror_reflection_func(ori, rand_1, rand_2):
        ori[2] = -ori[2]
        return ori
    
    @staticmethod
    def lambertian_reflection_func(ori, rand_1, rand_2):
        phi = rand_2 * 2 * np.pi

        cos_t = -np.sqrt(rand_1) # cosine-weighted hemisphere sampling, minus because surface has the biggest height
        sin_t = np.sqrt(1 - rand_1)
        cos_p = np.cos(phi)
        sin_p = np.sin(phi)

        return orientation(cos_t, sin_t, cos_p, sin_p)
    
    @staticmethod
    def uniform_reflection_func(ori, rand_1, rand_2):
        theta = rand_1 * np.pi / 2
        phi = rand_2 * 2 * np.pi

        cos_t = -np.cos(theta) # uniform sampling
        sin_t = np.sin(theta)
        cos_p = np.cos(phi)
        sin_p = np.sin(phi)

        return orientation(cos_t, sin_t, cos_p, sin_p)
    
class MirrorReflection(SurfaceReflection):
    def __init__(self):
        super().__init__(self.mirror_reflection_func)

class LambertianReflection(SurfaceReflection):
    def __init__(self):
        super().__init__(self.lambertian_reflection_func)

class UniformReflection(SurfaceReflection):
    def __init__(self):
        super().__init__(self.uniform_reflection_func)