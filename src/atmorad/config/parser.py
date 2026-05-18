import tomllib
import copy
from pathlib import Path

from atmorad.environment.atmosphere import AtmosphericMedium, AtmosphericLayer
from atmorad.environment.surface import SurfaceMaterial, SplitHalfXMap, CircleMap, CheckerboardMap, UniformMap, FlatSurface
from .models import (
    MetadataConfig, EngineConfig, SourceConfig, GeometryConfig, OutputConfig, DetectorConfig, SimConfig
)
from atmorad.physics import SCATTERING_MODELS, REFLECTION_MODELS

CURRENT_DIR = Path(__file__).parent
DEFAULT_CONFIG_PATH = CURRENT_DIR / "default_config.toml"

def _deep_merge_dicts(base: dict, override: dict) -> dict:
    base_copy = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and key in base_copy and isinstance(base_copy[key], dict):
            base_copy[key] = _deep_merge_dicts(base_copy[key], value)
        else:
            base_copy[key] = copy.deepcopy(value)
    return base_copy

def _parse_atmosphere_materials(materials_data: dict) -> dict[str, AtmosphericMedium]:
    parsed = {}
    for name, properties in materials_data.items():
        scat_data = properties.pop("scattering")
        scat_type = scat_data.pop("type")
        phase_function = SCATTERING_MODELS[scat_type](**scat_data)
        
        parsed[name] = AtmosphericMedium(
            extinction_coeff=properties["extinction_coeff_per_km"],
            ssa=properties["ssa"],
            phase_function=phase_function
        )
    return parsed

def _parse_surface(surface_config: dict, materials_config: dict, geom_config: GeometryConfig) -> FlatSurface:
    parsed_materials = {}
    for material_name, material_data in materials_config.items():
        ref_data = material_data.pop("reflection")
        ref_type = ref_data.pop("type")
        reflection_model = REFLECTION_MODELS[ref_type](**ref_data)
        
        parsed_materials[material_name] = SurfaceMaterial(
            albedo=material_data["albedo"],
            reflection=reflection_model
        )

    map_name = surface_config.get("name")
    is_periodic = (geom_config.boundary_condition == "periodic")
    
    if map_name == "uniform":
        ground_map = UniformMap()
        ordered_materials = [parsed_materials[surface_config["material"]]]
    elif map_name == "split-half-x":
        ground_map = SplitHalfXMap()
        ordered_materials = [
            parsed_materials[surface_config["material_left"]],
            parsed_materials[surface_config["material_right"]]
        ]
    elif map_name == "circle":
        ground_map = CircleMap(radius_km=surface_config["radius_km"])
        ordered_materials = [
            parsed_materials[surface_config["material_in"]],
            parsed_materials[surface_config["material_out"]]
        ]
    elif map_name == "checkerboard":
        ground_map = CheckerboardMap(tile_size_km=surface_config["tile_size_km"])
        ordered_materials = [
            parsed_materials[surface_config["material_a"]],
            parsed_materials[surface_config["material_b"]]
        ]
    else:
        raise ValueError(f"Unsupported surface map name: '{map_name}'")

    return FlatSurface(ground_map=ground_map, ground_types=ordered_materials, 
                       domain_x_km=geom_config.domain_size_x_km, domain_y_km=geom_config.domain_size_y_km, is_periodic=is_periodic)

def _parse_layers(raw_layers: list, atm_materials: dict[str, AtmosphericMedium]) -> list[AtmosphericLayer]:
    parsed_layers = []
    for layer_data in raw_layers:
        thickness = layer_data["z_range_km"][1] - layer_data["z_range_km"][0]
        
        components = []
        for material in layer_data["materials"]:
            medium_obj = atm_materials[material["type"]]
            weight = material["weight"]
            components.append((medium_obj, weight))
            
        parsed_layers.append(AtmosphericLayer(thickness=thickness, components=components))
    return parsed_layers

def load_config(custom_config_path: Path) -> tuple[SimConfig, FlatSurface, list[AtmosphericLayer]]:
    with open(DEFAULT_CONFIG_PATH, "rb") as f:
        default_config_data = tomllib.load(f)
    
    with open(custom_config_path, "rb") as f:
        custom_config_data = tomllib.load(f)
        
    config_data = _deep_merge_dicts(default_config_data, custom_config_data)

    metadata = MetadataConfig(**config_data["metadata"])
    engine = EngineConfig(**config_data["engine"])
    source = SourceConfig(**config_data["source"])
    geometry = GeometryConfig(**config_data["geometry"])
    output = OutputConfig(**config_data["output"])
    detectors = DetectorConfig(**config_data["detectors"])

    atm_materials = _parse_atmosphere_materials(config_data.get("atmosphere_materials", {}))
    layers = _parse_layers(config_data["layer"], atm_materials)
    
    surface = _parse_surface(
        config_data.get("surface", {}), 
        config_data.get("surface_materials", {}),
        geometry
    )

    return SimConfig(
        metadata=metadata,
        engine=engine,
        source=source,
        geometry=geometry,
        output=output,
        detectors=detectors,
    ), surface, layers