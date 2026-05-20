from pathlib import Path

import pytest

from atmorad.engine import MCRadiationRunner

CONFIG_DIR = Path(__file__).parent / "configs"
CONFIG_FILES = list(str(filename) for filename in CONFIG_DIR.glob("*.toml"))


@pytest.mark.parametrize(
    "sim_context",
    CONFIG_FILES,
    indirect=True,
    ids=CONFIG_FILES,
)
def test_energy_conservation(sim_context):
    runner = MCRadiationRunner(sim_context)
    runner.run()
    results = runner.get_results()
    reflected = results.get("photons_escaped_toa", 0)
    transmitted = results.get("photons_absorbed_surface", 0)
    absorbed_atm = results.get("photons_absorbed_atmosphere", 0)

    total_out = reflected + transmitted + absorbed_atm

    assert total_out == sim_context.config.engine.num_photons


@pytest.mark.parametrize(
    "sim_context",
    CONFIG_FILES,
    indirect=True,
    ids=CONFIG_FILES,
)
def test_no_nan_in_maps(sim_context):
    runner = MCRadiationRunner(sim_context)
    runner.run()
    results = runner.get_results()

    if "surface_flux_map_2d" in results:
        import numpy as np

        assert not np.isnan(results["surface_flux_map_2d"]).any()
