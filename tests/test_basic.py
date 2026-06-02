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
        if "energy_budget_energy_outgoing_toa" in ds:
            reflected = float(ds["energy_budget_energy_outgoing_toa"].values)
            transmitted = float(ds["energy_budget_energy_absorbed_surface"].values)
            absorbed_atm = float(ds["energy_budget_energy_absorbed_atmosphere"].values)
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
        surface_map_key = "surface_absorption_map_surface_absorption_map_2d"

        if surface_map_key in ds:
            assert not np.isnan(ds[surface_map_key].values).any()

        if "flux_maps_incident_flux_down_3d" in ds:
            assert not np.isnan(ds["flux_maps_incident_flux_down_3d"].values).any()

        if "flux_maps_incident_flux_up_3d" in ds:
            assert not np.isnan(ds["flux_maps_incident_flux_up_3d"].values).any()
