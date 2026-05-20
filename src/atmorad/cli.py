import argparse
import logging
import sys
import traceback
from pathlib import Path

from atmorad.builder import build_context
from atmorad.output import DataIO, ResultAnalyzer

from atmorad.engine import MCRadiationRunner


def main():
    parser = argparse.ArgumentParser(prog="atmorad")
    parser.add_argument("config", help="path to the simulation config TOML file", type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-v", "--verbose", action="store_true", help="increase output verbosity")
    group.add_argument("-q", "--quiet", action="store_true", help="suppress all output")
    args = parser.parse_args()

    if args.quiet:
        logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")
    elif args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        if not args.config.exists():
            raise FileNotFoundError(f"Configuration file '{args.config}' not found.")

        config_path = args.config.resolve()
        if args.verbose:
            logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
        else:
            logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")

        logging.info(f"Loading configuration from: {config_path.name}...")
        context = build_context(config_path)

        logging.info("Generating output directory...")
        data_io = DataIO(context.config)

        logging.info(
            f"Starting {context.config.engine.cpu_cores}-core simulation ({context.config.engine.num_photons} photons)..."
        )
        runner = MCRadiationRunner(context)
        runner.run()

        results_dict = runner.get_results()
        analyzer = ResultAnalyzer(results_dict, context.config)

        if not args.quiet:
            print("\n" + analyzer.summary())

        logging.info("Saving results to disk...")
        data_io.save_metadata(context.config, results_dict)
        data_io.save_results(results_dict)

        if context.config.output.save_plots:
            logging.info("Generating and saving plots...")
            data_io.save_all_artifacts(analyzer, results_dict)

        logging.info("Done! Simulation artifacts saved successfully.")

    except KeyboardInterrupt:
        print("\nSimulation aborted by user.")
        sys.exit(130)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")

        if args.verbose:
            print("\n--- Detailed Stack Trace ---")
            traceback.print_exc()
        else:
            print("\n(Run with --verbose to see the full stack trace for debugging)")

        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
