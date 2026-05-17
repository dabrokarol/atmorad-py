from atmorad.physics.geometry import orientation, rotate, sun_zenith_to_direction
from atmorad.physics import scattering, reflection

class AtmosphereScatterings:
    HenyeyGreenstein = scattering.HenyeyGreensteinScattering
    Uniform = scattering.IsotropicScattering

class SurfaceReflections:
    MirrorReflection = reflection.MirrorReflection
    LambertianReflection = reflection.LambertianReflection
    UniformReflection = reflection.UniformReflection