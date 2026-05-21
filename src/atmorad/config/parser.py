import copy
from pathlib import Path

import tomllib

from .models import (
    DetectorConfig,
    EngineConfig,
    EnvironmentConfig,
    GeometryConfig,
    MetadataConfig,
    OutputConfig,
    SimConfig,
    SourceConfig,
)

CURRENT_DIR = Path(__file__).parent
DEFAULT_CONFIG_PATH = CURRENT_DIR / "defaults.toml"


def _deep_merge_dicts(base: dict, override: dict) -> dict:
    base_copy = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and key in base_copy and isinstance(base_copy[key], dict):
            base_copy[key] = _deep_merge_dicts(base_copy[key], value)
        else:
            base_copy[key] = copy.deepcopy(value)
    return base_copy


def load_config(custom_config_path: Path) -> SimConfig:
    with open(DEFAULT_CONFIG_PATH, "rb") as f:
        default_config_data = tomllib.load(f)

    with open(custom_config_path, "rb") as f:
        custom_config_data = tomllib.load(f)

    config_data = _deep_merge_dicts(default_config_data, custom_config_data)

    env_config = EnvironmentConfig(
        atmosphere_materials=config_data.get("atmosphere_materials", {}),
        layers=config_data.get("layer", []),
        surface=config_data.get("surface", {}),
        surface_materials=config_data.get("surface_materials", {}),
        geometry=GeometryConfig(**config_data["geometry"]),
    )

    config = SimConfig(
        metadata=MetadataConfig(**config_data["metadata"]),
        engine=EngineConfig(**config_data["engine"]),
        source=SourceConfig(**config_data["source"]),
        output=OutputConfig(**config_data["output"]),
        detectors=DetectorConfig(**config_data["detectors"]),
        environment=env_config,
    )

    return config
