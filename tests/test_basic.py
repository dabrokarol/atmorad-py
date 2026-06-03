from pathlib import Path

import numpy as np
import pytest

from atmorad.api import run

CONFIG_DIR = Path(__file__).parent / "configs"
CONFIG_FILES = list(CONFIG_DIR.glob("*.toml"))


@pytest.mark.parametrize(
    "config_path",
    CONFIG_FILES,
    ids=lambda p: p.name,
)
def test_energy_conservation(config_path):
    results = run(config_path, quiet=True)
    if not isinstance(results, list):
        results = [results]

    for ds in results:
        if "energy_toa_outgoing" in ds:
            reflected = ds["energy_toa_outgoing"].item()
            transmitted = ds["energy_surface_absorbed"].item()
            absorbed_atm = ds["energy_atmosphere_absorbed"].item()
        else:
            reflected = transmitted = absorbed_atm = 0.0

        total_energy_out = reflected + transmitted + absorbed_atm
        expected_energy_in = float(ds.attrs.get("num_photons", 0))

        assert total_energy_out == pytest.approx(expected_energy_in, rel=1e-5), (
            f"Energy mismatch: Out({total_energy_out}) != In({expected_energy_in})"
        )


@pytest.mark.parametrize(
    "config_path",
    CONFIG_FILES,
    ids=lambda p: p.name,
)
def test_no_nan_in_maps(config_path):
    results = run(config_path, quiet=True)
    if not isinstance(results, list):
        results = [results]

    for ds in results:
        if "surface_absorption_map" in ds:
            assert not np.isnan(ds["surface_absorption_map"].values).any()

        if "downward_flux" in ds:
            assert not np.isnan(ds["downward_flux"].values).any()

        if "upward_flux" in ds:
            assert not np.isnan(ds["upward_flux"].values).any()
