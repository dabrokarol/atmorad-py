"""AtmoRad.py - a small usage example

This script demonstrates basic simulation usage
using the simplest possible setup (one layer of air, uniform Lambertian ground).

To see more complex examples (e.g. mixed surface boundaries, multiple atmosphere layers, and cloudy layers)
and learn how to generate the plots shown in README.md, check the script inside the `examples/` directory.

Enjoy!
"""

import numpy as np

from atmorad.engine.runner import MCRadiation
from atmorad.environment.scene import Scene
from atmorad.environment.atmosphere import Atmosphere, AtmosphericLayer, AtmosphericMedium
from atmorad.environment.surface import Surface, SurfaceMaterial, ProceduralMap
from atmorad.physics import SurfaceReflections, AtmosphereScatterings
from atmorad.io.data_io import OutputHandler
from atmorad.config.config import SimConfig


def main():

    #############################
    # 1. SIMULATION PARAMETERS ##
    #############################
    config = SimConfig(
        theta_sun_deg=0,
        cpu_cores=1
    )
    
    ###################
    # 2. ATMOSPHERE ###
    ###################
    # format: AtmosphericMedium(optical_density_per_km, ssa, scattering_phase_function)
    air = AtmosphericMedium(0.01, 0.5, AtmosphereScatterings.HenyeyGreenstein(g=0.5))

    # format: AtmosphericLayer(thickness_km, [(medium0, fraction0), ...])
    layer0 = AtmosphericLayer(20, air)

    atm = Atmosphere([layer0])

    ###################
    # 3. SURFACE ######
    ###################
    # format: SurfaceMaterial(albedo, reflection_object)
    material0 = SurfaceMaterial(0.5, SurfaceReflections.LambertianReflection())

    ground_map = ProceduralMap(ProceduralMap.uniform_ground)
    surface = Surface(ground_map, [material0])

    ###############################
    # 4. SCENE AND SIMULATION #####
    ###############################
    scene = Scene(surface, atm)
    sim = MCRadiation(config, scene)

    sim.run()

    ####################
    # 5. OUTPUTS #######
    ####################
    # generate a plot
    res = sim.get_results()
    fig_flux = res.plot_flux_profile()

    # save outputs to 'results' directory, overwrite=False will create a directory 'refults_timestamp'
    handler = OutputHandler('results', overwrite=True)
    handler.save_plot(fig_flux, 'vertical_flux_profile.png')

    # print basic results to the terminal
    handler.print_results(res)

    # save raw simulation data for later
    handler.save_metadata(config, res)
    handler.save_results(res)

if __name__ == '__main__':
    main()