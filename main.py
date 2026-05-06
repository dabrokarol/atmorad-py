import time

from src.simulation import MCRadiation
from src.scene import Scene, Space
from src.atmosphere import Atmosphere, AtmosphericLayer, AtmosphericMedium
from src.surface import Surface, SurfaceMaterial, ProceduralMap, GridMap
from src.physics import SurfaceReflections, AtmosphereScatterings
from src.results import Results
from src.data_io import read_config, read_results, OutputHandler

if __name__ == '__main__':
    config = read_config()
    
    # 1. ATMOSPHERE
    # Format: AtmosphericMedium(optical_density_per_km, ssa, scattering_phase_function)
    air = AtmosphericMedium(0.01, 0.5, AtmosphereScatterings.HenyeyGreenstein(g=0.5))
    clouds = AtmosphericMedium(5, 0.999999, AtmosphereScatterings.HenyeyGreenstein(g=0.85))

    # Format: AtmosphericLayer(height_km, [(medium0, fraction0), (medium1, fraction1), ...])
    # fractions should sum up to 1.0
    # Eg. layer = AtmosphericLayer(50, [(air, 0.3), (clouds, 0.7)]) 
    layer0 = AtmosphericLayer(5, [(air, 1)])
    layer1 = AtmosphericLayer(2, [(clouds, 1)])
    layer2 = AtmosphericLayer(10, [(air, 1)])
    atm = Atmosphere([layer0, layer1, layer2])

    # 2. SURFACE
    # Format: SurfaceMaterial(albedo, reflection_object)
    material0 = SurfaceMaterial(1, SurfaceReflections.MirrorReflection())
    material1 = SurfaceMaterial(0, SurfaceReflections.LambertianReflection())

    # Procedural map takes in a function that takes in a np.array of shape (2, N) or (3, N) and outputs array of shape (N) with integers
    # The returned integer array represents material IDs.
    ground_map = ProceduralMap(ProceduralMap.circle)
    surface = Surface(ground_map, [material0, material1])

    # 3. SCENE AND SIMULATION
    space = Space() # For now an empty object, represents end of atmosphere
    scene = Scene(surface, atm, space, config)

    sim = MCRadiation(config, scene)
    start_time = time.time_ns()
    sim.run()
    end_time = time.time_ns()
    # 4. OUTPUTS
    res = sim.get_results()
    fig_surf = res.surface_plot()
    fig_paths = res.plot_paths()

    handler = OutputHandler('results', overwrite=True)
    handler.save_plot(fig_paths, 'paths.png')
    handler.save_plot(fig_surf, 'surface.png')
    handler.save_metadata(config, (end_time - start_time) / 1e9)
    handler.save_results(res)
    handler.print_results(res)
