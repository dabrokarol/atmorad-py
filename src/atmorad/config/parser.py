import copy
from pathlib import Path

import tomllib

from atmorad.constants import ACCEPTED_EXTENSIONS

from .models import (
    SimConfig,
)

CURRENT_DIR = Path(__file__).parent
MATERIALS_CONFIG_PATH = CURRENT_DIR / "materials.toml"


def _deep_merge_dicts(base: dict, override: dict) -> dict:
    base_copy = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and key in base_copy and isinstance(base_copy[key], dict):
            base_copy[key] = _deep_merge_dicts(base_copy[key], value)
        else:
            base_copy[key] = copy.deepcopy(value)
    return base_copy


def load_config(custom_config_path: Path) -> SimConfig:
    if not custom_config_path.exists():
        raise FileNotFoundError(f"Config file not found at {custom_config_path}")

    if custom_config_path.suffix.lower() not in ACCEPTED_EXTENSIONS:
        raise ValueError(
            f"Invalid configuration file extension: {custom_config_path.suffix}"
            f"Allowed extensions: {', '.join(ACCEPTED_EXTENSIONS)}"
        )

    with open(MATERIALS_CONFIG_PATH, "rb") as f:
        materials_config_data = tomllib.load(f)

    with open(custom_config_path, "rb") as f:
        custom_config_data = tomllib.load(f)

    config_data = _deep_merge_dicts(materials_config_data, custom_config_data)

    env_keys = ["atmosphere_materials", "surface_materials", "surface", "geometry"]
    environment_data = {key: config_data.pop(key, {}) for key in env_keys}
    environment_data["layers"] = config_data.pop("layer", [])
    config_data["environment"] = environment_data
    config_data["config_path"] = custom_config_path

    return SimConfig(**config_data)
