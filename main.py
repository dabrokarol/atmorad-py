import tomllib
import numpy as np
from src.simulation import MCRadiation
from src.scene import Scene, Atmosphere, Surface, AtmosphericLayer, AtmosphericMedium, SurfaceMaterial, ProceduralMap, Scattering, SurfaceReflection, Space

def read_config(path = 'config.toml'):
    try:
        with open(path, 'rb') as f:
            data = tomllib.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"config file missing at {path}")
    return data

if __name__ == '__main__':
    rng = np.random.default_rng(42)
    config = read_config()
    
    air = AtmosphericMedium(1, 0.5, Scattering(Scattering.henyey_greenstein, 0.1, 1000))
    clouds = AtmosphericMedium(20, 0.9, Scattering(Scattering.henyey_greenstein, 0.8, 1000))
    layer1 = AtmosphericLayer(10, [(air, 0.7), (clouds, 0.3)])
    layer2 = AtmosphericLayer(10, [(air, 1)])
    atm = Atmosphere([layer1, layer2])

    mirror = SurfaceMaterial(1, SurfaceReflection(SurfaceReflection.mirror_reflection))
    diffuse = SurfaceMaterial(0.7, SurfaceReflection(SurfaceReflection.lambertian_reflection))
    map = ProceduralMap(ProceduralMap.checkerboard)
    surf = Surface(map, [mirror, diffuse])

    space = Space()

    scene = Scene(surf, atm, space, config)

    sim = MCRadiation(config['simulation'], scene, rng)

    sim.run()

    sim.print_results()

    sim.plot_paths()
