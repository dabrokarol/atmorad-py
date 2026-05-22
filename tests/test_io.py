from pathlib import Path

import netCDF4 as nc
import numpy as np
import pytest

from atmorad.engine import MCRadiationRunner
from atmorad.output import DataIO

CONFIG_DIR = Path(__file__).parent / "configs"
CONFIG_FILES = list(str(filename) for filename in CONFIG_DIR.glob("*.toml"))


def test_netcdf_dictionary_reconstruction(tmp_path):
    """Checks whether DataIO properly handles saving .nc files."""
    test_file = tmp_path / "test_data.nc"

    original_data = {
        "temperature": np.array([290.5, 291.0, 295.2]),
        "grid": {
            "x": np.array([0, 1, 2]),
            "y": np.array([0, 1, 2]),
            "resolution": 1.5,
            "domain": "global",
        },
        "iterations": 1000,
    }

    with nc.Dataset(test_file, "w", format="NETCDF4") as ncfile:
        DataIO._save_dict_to_group(ncfile, original_data)

    with nc.Dataset(test_file, "r") as ncfile:
        loaded_data = DataIO._load_group_to_dict(ncfile)

    assert loaded_data["iterations"] == 1000
    assert loaded_data["grid"]["domain"] == "global"
    assert loaded_data["grid"]["resolution"] == 1.5

    np.testing.assert_array_equal(loaded_data["temperature"], original_data["temperature"])
    np.testing.assert_array_equal(loaded_data["grid"]["x"], original_data["grid"]["x"])


def assert_dicts_close(dict1, dict2, path="", ignore_keys=None):
    """Recursively compares dictionaries containing numpy arrays, floats, and nested dicts."""

    keys1_str = {str(k) for k in dict1.keys()}
    keys2_str = {str(k) for k in dict2.keys()}

    assert keys1_str == keys2_str, f"Key mismatch at '{path}': {keys1_str} vs {keys2_str}"

    for k1 in dict1.keys():
        k2 = k1 if k1 in dict2 else str(k1)

        v1 = dict1[k1]
        v2 = dict2[k2]
        current_path = f"{path}/{k1}" if path else str(k1)

        if isinstance(v1, dict):
            assert isinstance(v2, dict), (
                f"Type mismatch at '{current_path}': {type(v1)} vs {type(v2)}"
            )
            assert_dicts_close(v1, v2, current_path, ignore_keys)

        elif isinstance(v1, np.ndarray) or isinstance(v2, np.ndarray):
            np.testing.assert_allclose(v1, v2, err_msg=f"Array mismatch at '{current_path}'")

        elif isinstance(v1, (float, np.floating)):
            assert np.isclose(v1, v2), f"Float value mismatch at '{current_path}': {v1} != {v2}"

        else:
            assert v1 == v2, f"Value mismatch at '{current_path}': {v1} != {v2}"

    return True


@pytest.mark.parametrize(
    "sim_context",
    CONFIG_FILES,
    indirect=True,
    ids=CONFIG_FILES,
)
def test_data_io_save_load_sim(sim_context, tmp_path):
    config = sim_context.config
    config.output.path = tmp_path
    data_io = DataIO(config)

    sim = MCRadiationRunner(
        context=sim_context,
        quiet=True,
        on_checkpoint=data_io.save_checkpoint,
        on_finish=data_io.save_simulation_run,
        on_cleanup=data_io.delete_checkpoint,
        load_checkpoint_fn=data_io.load_checkpoint,
    )

    sim.run()
    results = sim.get_results()

    # --- Assert: Save/Load Simulation Run ---
    config.output.path = "results"
    saved_dir = tmp_path / config.metadata.experiment_name
    config_2, results_2 = data_io.load_simulation_data(saved_dir)

    # Normalize volatile fields for comparison
    config_2.output.path = config.output.path
    config_2.config_path = config.config_path

    assert config == config_2, "Loaded configuration does not match the original."
    assert_dicts_close(results, results_2)

    data_io.save_checkpoint(simulated_photons=42, results=results)
    photons, results_from_checkpoint, config_from_checkpoint = data_io.load_checkpoint()

    config_from_checkpoint.output.path = config.output.path
    config_from_checkpoint.config_path = config.config_path

    assert photons == 42, f"Expected 42 simulated photons, got {photons}"
    assert config == config_from_checkpoint, "Checkpoint configuration does not match the original."
    assert_dicts_close(results, results_from_checkpoint)
