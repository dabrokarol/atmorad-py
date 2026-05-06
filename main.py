import time
import numpy as np

from src.simulation import MCRadiation
from src.scene import Scene, Space
from src.atmosphere import Atmosphere, AtmosphericLayer, AtmosphericMedium
from src.surface import Surface, SurfaceMaterial, ProceduralMap
from src.physics import SurfaceReflections, AtmosphereScatterings
from src.data_io import OutputHandler
from src.config import SimConfig


def main():

    # 1. SIMULATION PARAMETERS
    config = SimConfig(
        num_photons=100_000,
        num_track=100,
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
    material0 = SurfaceMaterial(1, SurfaceReflections.MirrorReflection())
    material1 = SurfaceMaterial(0, SurfaceReflections.LambertianReflection())

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

    # 4. OUTPUTS
    res = sim.get_results()
    handler = OutputHandler('demo_outputs', overwrite=False)

    fig_surf = res.surface_flux_plot()
    fig_paths = res.plot_paths()
    fig_flux = res.plot_flux_profile()
    fig_scat_hist = res.plot_scattering_histogram()

    handler.save_plot(fig_paths, 'sample_paths.png')
    handler.save_plot(fig_surf, 'surface_flux_map.png')
    handler.save_plot(fig_flux, 'flux.png')
    handler.save_plot(fig_scat_hist, 'scattering_counts.png')

    handler.save_metadata(config, (end_time - start_time) / 1e9)
    handler.save_results(res)
    handler.print_results(res)

if __name__ == '__main__':
    main()