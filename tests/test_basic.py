from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from atmorad.config.models import AtmosphereMaterialConfig, SurfaceMaterialConfig
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
    
    fate_res = results.detectors.get("fate") or results.detectors.get("FateDetector")
    
    if fate_res:
        reflected = fate_res.photons_escaped_toa
        transmitted = fate_res.photons_absorbed_surface
        absorbed_atm = fate_res.photons_absorbed_atmosphere
    else:
        reflected = transmitted = absorbed_atm = 0

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

    # Check Boundary Maps (Surface Absorption)
    boundary_res = results.detectors.get("boundary_flux") or results.detectors.get("BoundaryAbsorptionDetector")
    if boundary_res and boundary_res.surface_absorption_map_2d is not None:
        assert not np.isnan(boundary_res.surface_absorption_map_2d).any()
        
    # Check Boundary Maps (TOA reflection)
    if boundary_res and boundary_res.toa_flux_map_2d is not None:
        assert not np.isnan(boundary_res.toa_flux_map_2d).any()

    # Check Incident Plane Maps (Downward and Upward)
    plane_res = results.detectors.get("plane_flux") or results.detectors.get("IncidentFluxMapDetector")
    if plane_res:
        for flux_map in plane_res.incident_flux_down_maps_2d.values():
            assert not np.isnan(flux_map).any()
        for flux_map in plane_res.incident_flux_up_maps_2d.values():
            assert not np.isnan(flux_map).any()


def test_invalid_reflection_model_raises_error():
    with pytest.raises(ValidationError) as exc_info:
        SurfaceMaterialConfig(albedo=0.8, reflection={"type": "made_up_magic_mirror"})

    assert "Surface reflection model not found" in str(exc_info.value)


def test_invalid_scattering_model_raises_error():
    with pytest.raises(ValidationError) as exc_info:
        AtmosphereMaterialConfig(
            extinction_coeff_per_km=0.1, ssa=0.9, scattering={"type": "quantum_scattering"}
        )

    assert "Scattering model not found" in str(exc_info.value)