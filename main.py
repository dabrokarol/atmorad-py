"""AtmoRad.py - a small usage example

This script demonstrates basic simulation usage
using the simplest possible setup (one layer of air, uniform Lambertian ground).

To see more complex examples (e.g. mixed surface boundaries, multiple atmosphere layers, and cloudy layers)
and learn how to generate the plots shown in README.md, check the script inside the `examples/` directory.

Enjoy!
"""

import time
import numpy as np

from atmorad.simulation import MCRadiation
from atmorad.scene import Scene
from atmorad.atmosphere import Atmosphere, AtmosphericLayer, AtmosphericMedium
from atmorad.surface import Surface, SurfaceMaterial, ProceduralMap
from atmorad.physics import SurfaceReflections, AtmosphereScatterings
from atmorad.data_io import OutputHandler
from atmorad.config import SimConfig


def main():

    #############################
    # 1. SIMULATION PARAMETERS ##
    #############################
    config = SimConfig(
        theta_sun_deg=60
    )
    
    ###################
    # 2. ATMOSPHERE ###
    ###################
    # format: AtmosphericMedium(optical_density_per_km, ssa, scattering_phase_function)
    air = AtmosphericMedium(0.01, 0.5, AtmosphereScatterings.HenyeyGreenstein(g=0.5))

    # format: AtmosphericLayer(thickness_km, [(medium0, fraction0), ...])
    layer0 = AtmosphericLayer(20, [(air, 1)])

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
    hmax = atm.get_total_thickness() # height of the eintire atmospheric layer
    flux_measures_z = np.arange(0, hmax, 0.5) # heights at which flux will be measured (0 - top of atmosphere)

    scene = Scene(surface, atm)
    sim = MCRadiation(config, scene, flux_measures_z)

    start_time = time.perf_counter_ns() # for mesuring time
    sim.run()
    end_time = time.perf_counter_ns()

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
    handler.save_metadata(config, (end_time - start_time) / 1e9)
    handler.save_results(res)

if __name__ == '__main__':
    main()