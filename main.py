import logging
import argparse
from pathlib import Path

from atmorad.engine.runner import MCRadiationRunner
from atmorad.data_io import DataIO
from atmorad.config import parse_config
from atmorad.analyzer import ResultAnalyzer
from atmorad.environment import Scene, Atmosphere

def main():
    parser = argparse.ArgumentParser(prog="AtmoRad", usage="uv run main.py <path-to-config>")
    parser.add_argument("config", nargs="?", default="demo_config.toml", help="path to config TOML", type=Path)
    parser.add_argument("-v", "--verbose", action="store_true", help="increase output verbosity")
    args = parser.parse_args()

    config_path = Path.cwd() / args.config
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")

    logging.info(f"Loading configuration from: {config_path.name}...")
    context = parse_config(config_path)

    logging.info("Generating output directory...")
    data_io = DataIO(context.config)
    
    logging.info(f"Starting {context.config.engine.cpu_cores}-core simulation ({context.config.engine.num_photons} photons)...")
    runner = MCRadiationRunner(context)
    runner.run()

    results_dict = runner.get_results()
    analyzer = ResultAnalyzer(results_dict, context.config)

    print("\n" + analyzer.summary())

    logging.info("Saving results to disk...")
    
    data_io.save_metadata(context.config, results_dict)
    data_io.save_results(results_dict)
    
    if context.config.output.save_plots:
        logging.info("Generating and saving plots...")
        data_io.save_all_artifacts(analyzer, results_dict)

    logging.info("Done! Simulation artifacts saved successfully.")

if __name__ == '__main__':
    main()