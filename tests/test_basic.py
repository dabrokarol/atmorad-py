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
        if "fate_energy_outgoing_toa" in ds:
            reflected = float(ds["fate_energy_outgoing_toa"].values)
            transmitted = float(ds["fate_energy_absorbed_surface"].values)
            absorbed_atm = float(ds["fate_energy_absorbed_atmosphere"].values)
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
        if "surface_absorption_surface_absorption_map_2d" in ds:
            assert not np.isnan(ds["surface_absorption_surface_absorption_map_2d"].values).any()

        if "plane_flux_incident_flux_down_3d" in ds:
            assert not np.isnan(ds["plane_flux_incident_flux_down_3d"].values).any()

        if "plane_flux_incident_flux_up_3d" in ds:
            assert not np.isnan(ds["plane_flux_incident_flux_up_3d"].values).any()
