import dataclasses
from pathlib import Path

import numpy as np
import pytest

from atmorad.engine import MCRadiationRunner
from atmorad.output import DataIO

CONFIG_DIR = Path(__file__).parent / "configs"
CONFIG_FILES = list(str(filename) for filename in CONFIG_DIR.glob("*.toml"))


def assert_dicts_close(d1: dict, d2: dict, rtol=1e-5, atol=1e-8, path=""):
    assert isinstance(d1, dict) and isinstance(d2, dict)

    keys1, keys2 = set(d1.keys()), set(d2.keys())
    assert keys1 == keys2, (
        f"[{path}] Key mismatch! Missing in d1: {keys2 - keys1}, missing in d2: {keys1 - keys2}"
    )

    for k in d1:
        v1, v2 = d1[k], d2[k]
        current_path = f"{path}.{k}" if path else k

        if isinstance(v1, dict):
            assert_dicts_close(v1, v2, rtol, atol, current_path)

        elif isinstance(v1, np.ndarray) or isinstance(v2, np.ndarray):
            v1_arr, v2_arr = np.asarray(v1), np.asarray(v2)
            np.testing.assert_allclose(
                v1_arr,
                v2_arr,
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            )

        elif isinstance(v1, float) or isinstance(v2, float):
            assert np.isclose(v1, v2, rtol=rtol, atol=atol, equal_nan=True)

        else:
            assert v1 == v2


@pytest.mark.parametrize(
    "sim_context",
    CONFIG_FILES,
    indirect=True,
    ids=CONFIG_FILES,
)
def test_data_io_save_load_sim(sim_context, tmp_path):
    config = sim_context.config
    config.output.path = str(tmp_path)
    config.output.overwrite = True

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
    results = sim.results
    config_2, results_2 = DataIO.load_simulation_data(data_io.base_dir)

    config_2.output.path = config.output.path
    config_2.config_path = config.config_path

    assert config == config_2, "Loaded configuration does not match the original."

    assert_dicts_close(dataclasses.asdict(results), dataclasses.asdict(results_2))

    data_io.save_checkpoint(simulated_photons=42, results=results)
    photons, results_from_checkpoint, config_from_checkpoint = data_io.load_checkpoint()

    config_from_checkpoint.output.path = config.output.path
    config_from_checkpoint.config_path = config.config_path

    assert photons == 42, f"Expected 42 simulated photons, got {photons}"
    assert config == config_from_checkpoint, "Checkpoint configuration does not match the original."

    assert_dicts_close(dataclasses.asdict(results), dataclasses.asdict(results_from_checkpoint))
