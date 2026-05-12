from src.physics.geometry import orientation, rotate, sun_elevation_deg_to_direction, sun_elevation_rad_to_direction
from src.physics import scattering, reflection

class AtmosphereScatterings:
    HenyeyGreenstein = scattering.HenyeyGreenstein
    Uniform = scattering.Uniform

class SurfaceReflections:
    MirrorReflection = reflection.MirrorReflection
    LambertianReflection = reflection.LambertianReflection
    UniformReflection = reflection.UniformReflection