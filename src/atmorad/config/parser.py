import copy
from pathlib import Path

import tomllib

from atmorad.constants import ACCEPTED_EXTENSIONS

from .models import (
    SimConfig,
)

CURRENT_DIR = Path(__file__).parent


def _deep_merge_dicts(base: dict, scenario: dict) -> dict:
    base_copy = copy.deepcopy(base)
    for key, value in scenario.items():
        if isinstance(value, dict) and key in base_copy and isinstance(base_copy[key], dict):
            base_copy[key] = _deep_merge_dicts(base_copy[key], value)
        else:
            base_copy[key] = copy.deepcopy(value)
    return base_copy


def _build_single_config(raw_config_data: dict, config_path: Path) -> SimConfig:
    data = copy.deepcopy(raw_config_data)

    env_keys = ["atmosphere_materials", "surface_materials", "surface", "geometry"]
    environment_data = {key: data.pop(key, {}) for key in env_keys}
    environment_data["layers"] = data.pop("layer", [])
    data["environment"] = environment_data

    data["config_path"] = config_path

    return SimConfig(**data)


def load_scenarios(config_path: Path) -> list[SimConfig]:
    """Read TOML and returns SimConfig list (base or scenarios)."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    if config_path.suffix.lower() not in ACCEPTED_EXTENSIONS:
        raise ValueError(
            f"Invalid configuration file extension: {config_path.suffix}"
            f"Allowed extensions: {', '.join(ACCEPTED_EXTENSIONS)}"
        )

    with open(config_path, "rb") as f:
        raw_data = tomllib.load(f)

    scenarios = raw_data.pop("scenario", [])

    if "metadata" not in raw_data:
        raw_data["metadata"] = {}

    if not scenarios:
        return [_build_single_config(raw_data, config_path)]

    configs = []
    for idx, scenario_dict in enumerate(scenarios):
        scenario_name = scenario_dict.pop("name", None)

        base_copy = copy.deepcopy(raw_data)
        merged_raw = _deep_merge_dicts(base_copy, scenario_dict)

        if not scenario_name:
            raise ValueError(
                f"Configuration Error: Found an unnamed [[scenario]] (index {idx}).\n"
                f"Each scenario has to have an explicit name to ensure organized output files.\n"
                f"Example:\n"
                f"  [[scenario]]\n"
                f'  name = "base_scenario"'
            )

        merged_raw["metadata"]["scenario_name"] = scenario_name

        configs.append(_build_single_config(merged_raw, config_path))

    return configs
