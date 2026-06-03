import copy
import itertools
from pathlib import Path
from typing import Any

import tomllib

from atmorad.constants import ACCEPTED_EXTENSIONS

from .schemas import SimConfig


def _set_nested_value(data: dict, path: str, value: Any):
    keys = path.split(".")
    current = data
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def _deep_merge_dicts(base: dict, scenario: dict) -> dict:
    base_copy = copy.deepcopy(base)
    polymorphic_keys = ["type"]

    for key, value in scenario.items():
        if isinstance(value, dict) and key in base_copy and isinstance(base_copy[key], dict):
            base_dict = base_copy[key]
            other_dict = value

            type_changed = any(
                p_key in base_dict and p_key in other_dict and base_dict[p_key] != other_dict[p_key]
                for p_key in polymorphic_keys
            )

            if type_changed:
                base_copy[key] = copy.deepcopy(other_dict)
            else:
                base_copy[key] = _deep_merge_dicts(base_copy[key], other_dict)
        else:
            base_copy[key] = copy.deepcopy(value)
    return base_copy


def load_scenarios(config_path: str | Path) -> list[SimConfig]:
    """Wczytuje TOML i zwraca sekwencyjną listę konfiguracji."""
    path = Path(config_path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {path}")

    if path.suffix.lower() not in ACCEPTED_EXTENSIONS:
        raise ValueError(
            f"Invalid configuration file extension: {path.suffix}. "
            f"Allowed: {', '.join(ACCEPTED_EXTENSIONS)}"
        )

    with open(path, "rb") as f:
        raw_data = tomllib.load(f)

    scenarios = raw_data.pop("scenario", [])
    sweeps = raw_data.pop("sweep", [])

    if "metadata" not in raw_data:
        raw_data["metadata"] = {}

    base_configs_data = []
    if not scenarios:
        base_configs_data.append(raw_data)
    else:
        names = set()
        for idx, scenario_dict in enumerate(scenarios):
            scenario_name = scenario_dict.pop("name", None)
            if not scenario_name:
                raise ValueError(f"Configuration Error: Found unnamed [[scenario]] at index {idx}.")
            if scenario_name in names:
                raise ValueError(f"Overlapping scenario name: {scenario_name}")
            names.add(scenario_name)

            merged_raw = _deep_merge_dicts(raw_data, scenario_dict)
            merged_raw["metadata"]["scenario_name"] = scenario_name
            base_configs_data.append(merged_raw)

    if not sweeps:
        return [SimConfig(**data, config_path=path) for data in base_configs_data]

    final_configs = []
    sweep_params = [s["parameter"] for s in sweeps]
    sweep_values = [s["values"] for s in sweeps]

    for base_config in base_configs_data:
        base_name = base_config.get("metadata", {}).get("scenario_name", "baseline")

        # a cartesian product of value combinations
        for combo in itertools.product(*sweep_values):
            scenario_data = copy.deepcopy(base_config)

            name_parts = [base_name] if base_name != "baseline" else []

            for param_path, val in zip(sweep_params, combo):
                _set_nested_value(scenario_data, param_path, val)

                short_param = param_path.split(".")[-1]
                clean_val = str(val).replace(".", "_")
                name_parts.append(f"{short_param}_{clean_val}")

            scenario_data["metadata"]["scenario_name"] = (
                "_".join(name_parts) if name_parts else "baseline"
            )

            final_configs.append(SimConfig(**scenario_data, config_path=path))

    return final_configs
