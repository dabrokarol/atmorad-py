from pathlib import Path

from atmorad.config import EnvironmentConfig, GeometryConfig, load_config
from atmorad.environment import (
    Atmosphere,
    AtmosphericLayer,
    AtmosphericMedium,
    CheckerboardMap,
    CircleMap,
    FlatSurface,
    Scene,
    SplitHalfXMap,
    SurfaceMaterial,
    UniformMap,
)
from atmorad.physics.registry import REFLECTION_MODELS, SCATTERING_MODELS

from .engine.runner import SimContext


def _build_atmosphere_materials(materials_data: dict) -> dict[str, AtmosphericMedium]:
    built = {}
    for name, properties in materials_data.items():
        scat_data = properties.pop("scattering")
        scat_type = scat_data.pop("type")
        phase_function = SCATTERING_MODELS[scat_type](**scat_data)

        built[name] = AtmosphericMedium(
            extinction_coeff=properties["extinction_coeff_per_km"],
            ssa=properties["ssa"],
            phase_function=phase_function,
        )
    return built


def _build_surface(
    surface_config: dict, materials_config: dict, geom_config: GeometryConfig
) -> FlatSurface:
    built_materials = {}
    for material_name, material_data in materials_config.items():
        ref_data = material_data.pop("reflection")
        ref_type = ref_data.pop("type")
        reflection_model = REFLECTION_MODELS[ref_type](**ref_data)

        built_materials[material_name] = SurfaceMaterial(
            albedo=material_data["albedo"], reflection=reflection_model
        )

    map_name = surface_config.get("name")
    is_periodic = geom_config.boundary_condition == "periodic"

    if map_name == "uniform":
        ground_map = UniformMap()
        ordered_materials = [built_materials[surface_config["material"]]]
    elif map_name == "split-half-x":
        ground_map = SplitHalfXMap()
        ordered_materials = [
            built_materials[surface_config["material_left"]],
            built_materials[surface_config["material_right"]],
        ]
    elif map_name == "checkerboard":
        ground_map = CheckerboardMap(tile_size_km=surface_config["tile_size_km"])
        ordered_materials = [
            built_materials[surface_config["material_a"]],
            built_materials[surface_config["material_b"]],
        ]
    elif map_name == "circle":
        ground_map = CircleMap(radius_km=surface_config["radius_km"])
        ordered_materials = [
            built_materials[surface_config["material_in"]],
            built_materials[surface_config["material_out"]],
        ]
    else:
        raise ValueError(f"Unsupported surface map name: '{map_name}'")

    return FlatSurface(
        ground_map=ground_map,
        ground_types=ordered_materials,
        domain_x_km=geom_config.domain_size_x_km,
        domain_y_km=geom_config.domain_size_y_km,
        is_periodic=is_periodic,
    )


def _build_layers(
    raw_layers: list, atm_materials: dict[str, AtmosphericMedium]
) -> list[AtmosphericLayer]:
    built_layers = []
    for layer_data in raw_layers:
        thickness = layer_data["thickness_km"]
        components = [(atm_materials[m["type"]], m["weight"]) for m in layer_data["materials"]]
        built_layers.append(AtmosphericLayer(thickness=thickness, components=components))
    return built_layers


def _build_scene(env_config: EnvironmentConfig):
    atm_materials = _build_atmosphere_materials(env_config.atmosphere_materials)
    layers = _build_layers(env_config.layers, atm_materials)
    surface = _build_surface(env_config.surface, env_config.surface_materials, env_config.geometry)

    return Scene(surface, Atmosphere(layers))


def build_context(config_path: Path | str) -> SimContext:
    path = Path(config_path).resolve()

    config = load_config(path)

    scene = _build_scene(config.environment)

    return SimContext(config=config, scene=scene, config_path=path)
