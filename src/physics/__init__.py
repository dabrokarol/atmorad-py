from src.physics.geometry import orientation, rotate
from src.physics import scattering, reflection

class AtmosphereScatterings:
    HenyeyGreenstein = scattering.HenyeyGreenstein
    Uniform = scattering.Uniform

class SurfaceReflections:
    MirrorReflection = reflection.MirrorReflection
    LambertianReflection = reflection.LambertianReflection
    UniformReflection = reflection.UniformReflection