import tomllib


### OBJECTS USED TO CREATE AND RUN THE SIMULATION:
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
    ## Parse `config.toml`
    config = read_config()
    
    ## Create atmospheric mediums: AtmosphericMedium(optical_density_per_km, single_scattering_albedo, scattering_object)
    ## Scattering objects are kept in AtmosphereScatterings and take in assymetry parameter g as an argument upon creation
    air = AtmosphericMedium(0.001, 0.5, AtmosphereScatterings.HenyeyGreenstein(g=0.5))
    clouds = AtmosphericMedium(20, 0.999999, AtmosphereScatterings.HenyeyGreenstein(g=0.85))

    ## Compose layers with above definec materials: 
    ## AtmosphericLayer(height_km, [(medium1, percentage1), (medium2, percentage2), ...])
    ## Each layer consists of list of tuples: (medium, percentage); percentages should sum to 1
    ## Eg. layer = AtmosphericLayer(50, [(air, 0.3), (clouds, 0.7)]) 
    layer0 = AtmosphericLayer(50, [(air, 1)])
    layer1 = AtmosphericLayer(1, [(clouds, 1)])
    layer2 = AtmosphericLayer(50, [(air, 1)])

    ## Add layers to Atmosphere class
    ## Atmosphere([layer1, layer2, layer3])
    atm = Atmosphere([layer0, layer1, layer2])


    ## Create a surface material: SurfaceMaterial(albedo, reflection_object)
    ## Reflection objects are kept in SurfaceReflections class and should be called upon initialization eg:
    material0 = SurfaceMaterial(1, SurfaceReflections.MirrorReflection())
    material1 = SurfaceMaterial(0, SurfaceReflections.LambertianReflection())

    ## Create a map: 
    ## Procedural map takes in a function that takes in a np.array of shape (2, N) or (3, N) and outputs array of shape (N) with numbers
    ## You can test prefefined Procedural maps, eg. ProceduralMap(ProceduralMap.checkerboard)
    map = ProceduralMap(ProceduralMap.split_half_x)

    ## Surface is a class that keeps a map and list of surface materials
    surface = Surface(map, [material0, material1])

    ## For now an empty object, does nothing
    space = Space()

    ## Scene keeps information about simulation environment, create as follows:
    ## Scene(surface, atmosphere, space, config)
    scene = Scene(surface, atm, space, config)

    ## Simulation is an object that takes in config and scene
    sim = MCRadiation(config, scene)

    ## Run simulation
    sim.run()

    ## USEFUL FUNCTIONS:
    ## Print results to command line
    sim.print_results()
    ## Plot paths of `n_track` photons
    sim.plot_paths()
