from pathlib import Path

import numpy as np
import pytest

from atmorad.engine import MCRadiationRunner
from atmorad.models.results import FateResult, IncidentFluxMapResult, SurfaceAbsorptionResult

CONFIG_DIR = Path(__file__).parent / "configs"
CONFIG_FILES = list(str(filename) for filename in CONFIG_DIR.glob("*.toml"))


@pytest.mark.parametrize(
    "sim_context_list",
    CONFIG_FILES,
    indirect=True,
    ids=CONFIG_FILES,
)
def test_energy_conservation(sim_context_list):
    for sim_context in sim_context_list:
        runner = MCRadiationRunner(sim_context, quiet=True)
        runner.run()
        results = runner.results

        fate_res = results.detector_results.get("fate")

        if fate_res:
            assert isinstance(fate_res, FateResult)
            reflected = fate_res.energy_outgoing_toa
            transmitted = fate_res.energy_absorbed_surface
            absorbed_atm = fate_res.energy_absorbed_atmosphere
        else:
            reflected = transmitted = absorbed_atm = 0.0

        total_energy_out = reflected + transmitted + absorbed_atm
        expected_energy_in = float(sim_context.config.engine.num_photons)

        assert total_energy_out == pytest.approx(expected_energy_in, rel=1e-5), (
            f"Energy mismatch: Out({total_energy_out}) != In({expected_energy_in})"
        )


@pytest.mark.parametrize(
    "sim_context_list",
    CONFIG_FILES,
    indirect=True,
    ids=CONFIG_FILES,
)
def test_no_nan_in_maps(sim_context_list):
    for sim_context in sim_context_list:
        runner = MCRadiationRunner(sim_context, quiet=True)
        runner.run()
        results = runner.results

        # Check Surface Absorption Map (2D Array)
        surf_res = results.detector_results.get("surface_absorption")
        assert isinstance(surf_res, SurfaceAbsorptionResult)
        if surf_res and surf_res.surface_absorption_map_2d is not None:
            assert not np.isnan(surf_res.surface_absorption_map_2d).any(), (
                "NaN values detected in surface absorption map"
            )

        # Check Incident Plane Maps (3D Arrays)
        plane_res = results.detector_results.get("plane_flux")
        if plane_res:
            assert isinstance(plane_res, IncidentFluxMapResult)
            if plane_res.incident_flux_down_3d is not None:
                assert not np.isnan(plane_res.incident_flux_down_3d).any(), (
                    "NaN values detected in downward incident flux 3D map"
                )

            if plane_res.incident_flux_up_3d is not None:
                assert not np.isnan(plane_res.incident_flux_up_3d).any(), (
                    "NaN values detected in upward incident flux 3D map"
                )
