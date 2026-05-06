"""AtmoRad.py - a small usage example

This script will demonstrate a basic usage of simulation 
using the simplest-possible setup (one layer of air, uniform Lambertian ground).

To see more complex ecample (e.g. mixed surface boundaries, multiple atmosphere layers, cloudy layers)
and how to generate plots seen in README.md, check the script inside  `examples/` directory 

Enjoy!
"""

import time
import numpy as np
import sys
from pathlib import Path

# a script that will allow nested copies of this file to run properly, you can ignore it 
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir if (_script_dir / 'src').exists() else _script_dir.parent
sys.path.append(str(_project_root))


from src.simulation import MCRadiation
from src.scene import Scene
from src.atmosphere import Atmosphere, AtmosphericLayer, AtmosphericMedium
from src.surface import Surface, SurfaceMaterial, ProceduralMap
from src.physics import SurfaceReflections, AtmosphereScatterings
from src.data_io import OutputHandler
from src.config import SimConfig


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
    flux_measures_z = np.arange(0, hmax, 0.5) # heights at which flux will be measured (0 - top of atmosphere))

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