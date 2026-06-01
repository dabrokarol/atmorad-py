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

    polymorphic_keys = ["name", "type"]  # in case of change, overwrite, not merge

    for key, value in scenario.items():
        if isinstance(value, dict) and key in base_copy and isinstance(base_copy[key], dict):
            base_dict = base_copy[key]
            other_dict = value

            type_changed = False
            for p_key in polymorphic_keys:
                if p_key in base_dict and p_key in other_dict:
                    if base_dict[p_key] != other_dict[p_key]:
                        type_changed = True
                        break

            if type_changed:
                base_copy[key] = copy.deepcopy(other_dict)
            else:
                base_copy[key] = _deep_merge_dicts(base_copy[key], other_dict)
        else:
            base_copy[key] = copy.deepcopy(value)
    return base_copy


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
        return [SimConfig(**raw_data, config_path=config_path)]

    names = set()

    configs = []
    for idx, scenario_dict in enumerate(scenarios):
        scenario_name = scenario_dict.pop("name", None)

        if not scenario_name:
            raise ValueError(
                f"Configuration Error: Found an unnamed [[scenario]] (index {idx}).\n"
                f"Each scenario has to have an explicit name to ensure organized output files.\n"
                f"Example:\n"
                f"  [[scenario]]\n"
                f'  name = "tau_20"'
            )
        if scenario_name in names:
            raise ValueError(
                f"Overlapping scenario names in an experiment. Each scenario name should be unique. Overlapping: {scenario_name}"
            )
        names.add(scenario_name)

        merged_raw = _deep_merge_dicts(raw_data, scenario_dict)

        merged_raw["metadata"]["scenario_name"] = scenario_name

        configs.append(SimConfig(**merged_raw, config_path=config_path))

    return configs
