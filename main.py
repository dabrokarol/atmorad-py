import tomllib
import numpy as np

from src.simulation import MCRadiation
from src.scene import Scene, Space
from src.atmosphere import Atmosphere, AtmosphericLayer, AtmosphericMedium
from src.surface import Surface, SurfaceMaterial, ProceduralMap, GridMap
from src.physics import SurfaceReflections, AtmosphereScatterings


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
    
    air = AtmosphericMedium(0.001, 0.5, AtmosphereScatterings.HenyeyGreenstein(g=0.5))
    clouds = AtmosphericMedium(20, 0.999999, AtmosphereScatterings.HenyeyGreenstein(g=0.85))
    layer1 = AtmosphericLayer(50, [(air, 1)])
    layer2 = AtmosphericLayer(1, [(clouds, 1)])
    layer3 = AtmosphericLayer(50, [(air, 1)])
    atm = Atmosphere([layer1, layer2, layer3])

    mirror = SurfaceMaterial(1, SurfaceReflections.MirrorReflection())
    diffuse = SurfaceMaterial(0, SurfaceReflections.LambertianReflection())
    map = ProceduralMap(ProceduralMap.split_half_x)
    surf = Surface(map, [mirror, diffuse])

    space = Space()

    scene = Scene(surf, atm, space, config)

    sim = MCRadiation(config['simulation'], scene, rng)

    sim.run()

    sim.print_results()

    sim.plot_paths()
