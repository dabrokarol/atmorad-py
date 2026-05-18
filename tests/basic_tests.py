from pathlib import Path
from atmorad.config.models import load_config
from atmorad.engine.runner import MCRadiationRunner

def test_energy_conservation():
    config = load_config(Path("default_config.toml"))
    config.engine.num_photons = 5000
    config.engine.batch_size = 5000
    runner = MCRadiationRunner(config)
    runner.run()
    results = runner.get_results()
    reflected = results.get("photons_escaped_toa", 0)
    transmitted = results.get("photons_absorbed_surface", 0)
    absorbed_atm = results.get("photons_absorbed_atmosphere", 0)
    
    total_out = reflected + transmitted + absorbed_atm

    assert total_out == 5000, f"Energy not conserved, 5000 photons started, but eded {total_out}."

def test_no_nan_in_maps():
    config = load_config(Path("default_config.toml"))
    config.engine.num_photons = 1000
    runner = MCRadiationRunner(config)
    runner.run()
    results = runner.get_results()

    if "surface_flux_map_2d" in results:
        import numpy as np
        assert not np.isnan(results["surface_flux_map_2d"]).any(), "NaN found in apsorption maps"