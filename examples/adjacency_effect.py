"""
Adjacency Effect Simulation:
This file shows how to set up adjacency effect simulation and generate plots seen in README.md

Simulation structure:
- Three atmospheric layers (air, clouds, air)
- Surface split on half (x=0)
    - x<0: albedo=0 (fully absorbent surface)
    - x>0: albedo=1 (fully reflective surface, lambertian reflection)
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

    # 1. SIMULATION PARAMETERS
    config = SimConfig(
        num_photons=1_000_000,
        num_track=200,
        random_seed=42,
        theta_sun_deg=60,
        phi_sun_deg=0
    )
    
    # 2. ATMOSPHERE
    # Format: AtmosphericMedium(optical_density_per_km, ssa, scattering_phase_function)
    air = AtmosphericMedium(0.01, 0.5, AtmosphereScatterings.HenyeyGreenstein(g=0.5))
    clouds = AtmosphericMedium(5, 0.999999, AtmosphereScatterings.HenyeyGreenstein(g=0.85))

    # Format: AtmosphericLayer(thickness_km, [(medium0, fraction0), (medium1, fraction1), ...])
    # fractions should sum up to 1.0
    # Eg. layer = AtmosphericLayer(50, [(air, 0.3), (clouds, 0.7)]) 
    layer0 = AtmosphericLayer(5, [(air, 1)])
    layer1 = AtmosphericLayer(2, [(clouds, 1)])
    layer2 = AtmosphericLayer(10, [(air, 1)])
    atm = Atmosphere([layer0, layer1, layer2])

    # 3. SURFACE
    # Format: SurfaceMaterial(albedo, reflection_object)
    material0 = SurfaceMaterial(0, SurfaceReflections.LambertianReflection())
    material1 = SurfaceMaterial(1, SurfaceReflections.LambertianReflection())

    # Procedural map takes in a function that takes in a np.array of shape (2, N) or (3, N) and outputs array of shape (N) with integers
    # The returned integer array represents material IDs.
    ground_map = ProceduralMap(ProceduralMap.split_half_x)
    surface = Surface(ground_map, [material0, material1])

    # 4. SCENE AND SIMULATION
    hmax = atm.get_total_thickness() # height of the eintire atmospheric layer
    flux_measures_z = np.arange(0, hmax, 0.5) # heights at which flux will be measured (0 - top of atmosphere)

    scene = Scene(surface, atm)
    sim = MCRadiation(config, scene, flux_measures_z)

    start_time = time.perf_counter_ns() # for mesuring time
    sim.run()
    end_time = time.perf_counter_ns()

    # 5. OUTPUTS
    res = sim.get_results()
    handler = OutputHandler('results', overwrite=True)

    fig_surf = res.surface_flux_plot(title='Downward flux near the ground on the border\n of absorbant and reflective surfaces (border on X=0)')
    fig_paths = res.plot_paths()
    fig_flux = res.plot_flux_profile()
    fig_scat_hist = res.plot_scattering_histogram()

    handler.save_plot(fig_paths, '3d_photon_paths.png')
    handler.save_plot(fig_surf, 'surface_flux_map.png')
    handler.save_plot(fig_flux, 'vertical_flux_profile.png')
    handler.save_plot(fig_scat_hist, 'scattering_hist.png')

    handler.save_metadata(config, (end_time - start_time) / 1e9)
    handler.save_results(res)
    handler.print_results(res)

if __name__ == '__main__':
    main()