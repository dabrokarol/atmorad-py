import logging
import argparse
from pathlib import Path

from atmorad.engine.runner import MCRadiationRunner
from atmorad.data_io import DataIO
from atmorad.config.parser import load_config
from atmorad.results import ResultAnalyzer

def main():
    parser = argparse.ArgumentParser(prog="AtmoRad", usage="uv run main.py <path-to-config>")
    parser.add_argument("config", nargs="?", default="default_config.toml", help="path to config TOML", type=Path)
    parser.add_argument("-v", "--verbose", action="store_true", help="increase output verbosity")
    args = parser.parse_args()

    config_path = Path.cwd() / args.config
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")

    logging.info(f"Loading configuration from: {config_path.name}...")
    config = load_config(config_path)
    
    logging.info("Generating output directory...")
    handler = DataIO(config)
    
    logging.info(f"Starting {config.engine.cpu_cores}-core simulation ({config.engine.num_photons} photons)...")
    runner = MCRadiationRunner(config)
    runner.run()

    results_dict = runner.get_results()
    analyzer = ResultAnalyzer(results_dict, config)

    print("\n" + analyzer.summary())

    logging.info("Saving results to disk...")
    
    if config.output.save_boundary_flux_maps:
        fig_map = analyzer.plot_surface_flux_map()
        if fig_map: handler.save_plot(fig_map, 'surface_flux_map.png')
        
    if config.output.save_vertical_profile:
        fig_flux = analyzer.plot_flux_profile()
        if fig_flux: handler.save_plot(fig_flux, 'vertical_flux_profile.png')
        
        fig_heat = analyzer.plot_heating_rate()
        if fig_heat: handler.save_plot(fig_heat, 'heating_profile.png')

    if config.output.save_photon_paths:
        fig_paths = analyzer.plot_paths()
        if fig_paths: handler.save_plot(fig_paths, '3d_photon_paths.png')

    handler.save_metadata(config, results_dict)
    handler.save_results(results_dict)
    logging.info("Done! Simulation artifacts saved successfully.")

if __name__ == '__main__':
    main()