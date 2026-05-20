import logging
import argparse
from pathlib import Path

from atmorad.engine.runner import MCRadiationRunner
from atmorad.data_io import DataIO
from atmorad.config import parse_config
from atmorad.analyzer import ResultAnalyzer

def main():
    parser = argparse.ArgumentParser(prog="atmorad", usage="uv run main.py <path-to-config>")
    parser.add_argument("config", help="path to the simulation config TOML file", type=Path)
    parser.add_argument("-v", "--verbose", action="store_true", help="increase output verbosity")
    args = parser.parse_args()
    
    if not args.config.exists():
        raise FileNotFoundError(f"Error: Configuration file '{args.config}' not found.")

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