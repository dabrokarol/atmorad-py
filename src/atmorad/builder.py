from pathlib import Path

from atmorad.config import EnvironmentConfig, GeometryConfig, load_scenarios
from atmorad.environment import (
    Atmosphere,
    AtmosphericLayer,
    AtmosphericMedium,
    FlatSurface,
    Scene,
    SurfaceMaterial,
)
from atmorad.registry import REFLECTION_MODELS, SCATTERING_MODELS, SURFACE_MAPS

from .models import SimContext


def _build_atmosphere_materials(materials_data: dict) -> dict[str, AtmosphericMedium]:
    built = {}
    for name, properties in materials_data.items():
        scat_data = properties.scattering
        scat_type = scat_data["type"]
        scat_kwargs = {k: v for k, v in scat_data.items() if k != "type"}
        phase_function = SCATTERING_MODELS[scat_type](**scat_kwargs)

        built[name] = AtmosphericMedium(
            extinction_coeff=properties.extinction_coeff_per_km,
            ssa=properties.ssa,
            phase_function=phase_function,
        )
    return built


def _build_surface(
    surface_config: dict, materials_config: dict, geom_config: GeometryConfig
) -> FlatSurface:
    built_materials = {}
    for material_name, material_data in materials_config.items():
        ref_data = material_data.reflection
        ref_type = ref_data["type"]
        ref_kwargs = {k: v for k, v in ref_data.items() if k != "type"}
        reflection_model = REFLECTION_MODELS[ref_type](**ref_kwargs)

        built_materials[material_name] = SurfaceMaterial(
            albedo=material_data.albedo, reflection=reflection_model
        )

    map_name = surface_config["name"]
    is_periodic = geom_config.boundary_condition == "periodic"

    map_data = SURFACE_MAPS[map_name]
    MapClass = map_data["class"]
    material_keys = map_data["material_keys"]

    material_names = [surface_config[key] for key in material_keys]

    kwargs = {k: v for k, v in surface_config.items() if k not in material_keys and k != "name"}

    ground_map = MapClass(**kwargs)

    ordered_materials = [built_materials[name] for name in material_names]

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
        thickness = layer_data.thickness_km
        components = [
            (atm_materials[material.type], material.weight) for material in layer_data.materials
        ]
        built_layers.append(AtmosphericLayer(thickness=thickness, components=components))
    return built_layers


def _build_scene(env_config: EnvironmentConfig):
    atm_materials = _build_atmosphere_materials(env_config.atmosphere_materials)
    layers = _build_layers(env_config.layers, atm_materials)
    surface = _build_surface(env_config.surface, env_config.surface_materials, env_config.geometry)

    return Scene(surface, Atmosphere(layers))


def build_context_list(config_path: Path | str) -> list[SimContext]:
    path = Path(config_path).resolve()

    config_list = load_scenarios(path)

    context_list = []
    for config in config_list:
        scene = _build_scene(config.environment)
        context_list.append(SimContext(config=config, scene=scene))

    return context_list
