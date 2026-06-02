from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from atmorad.config.loader import load_scenarios
from atmorad.config.schemas import SimConfig
from atmorad.environment import Scene
from atmorad.output.io import DataIO, normalize_dataset
from atmorad.runner import execute_simulation

CONFIG_DIR = Path(__file__).parent / "configs"
CONFIG_FILES = list(CONFIG_DIR.glob("*.toml"))


@pytest.mark.parametrize(
    "config_path",
    CONFIG_FILES,
    ids=lambda p: p.name,
)
def test_data_io_save_load_sim(config_path, tmp_path):
    config = load_scenarios(config_path)[0]
    config.output.base_dir = tmp_path
    config.output.overwrite = True

    data_io = DataIO(config)
    scene = Scene.from_config(config)

    results_ds = execute_simulation(
        config=config,
        scene=scene,
        quiet=True,
        on_checkpoint=data_io.save_checkpoint,
    )

    data_io.save_simulation_run(results_ds)

    loaded_ds = data_io.load_checkpoint()
    assert loaded_ds is not None

    loaded_config = SimConfig.model_validate_json(loaded_ds.attrs["_simulation_config"])
    loaded_config.config_path = config.config_path
    loaded_config.output.base_dir = config.output.base_dir
    assert config == loaded_config

    expected_normalized = normalize_dataset(results_ds)
    xr.testing.assert_allclose(expected_normalized, loaded_ds)

    (data_io.base_dir / data_io.results_filename).unlink(missing_ok=True)
    results_ds.attrs["num_photons"] = 42
    data_io.save_checkpoint(results_ds)

    results_from_checkpoint = data_io.load_checkpoint()
    assert results_from_checkpoint is not None
    assert np.isclose(results_from_checkpoint.attrs["num_photons"], 42)

    xr.testing.assert_allclose(results_ds, results_from_checkpoint)
