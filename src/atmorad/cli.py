import argparse
import importlib.metadata
import importlib.resources as pkg_resources
import logging
import sys
import traceback
from pathlib import Path

from atmorad.builder import build_context
from atmorad.engine import MCRadiationRunner
from atmorad.output import DataIO, ResultAnalyzer


def init_config():
    out_path = Path("simulation.toml")

    if out_path.exists():
        print(
            f"Error: '{out_path}' already exists. Delete if you want to reinitialize.",
            file=sys.stderr,
        )
        sys.exit(1)

    config_data = pkg_resources.files("atmorad.config").joinpath("simulation.toml").read_text()
    out_path.write_text(config_data)
    print(f"Generated default configuration file at {out_path.resolve()}")


def main():
    try:
        __version__ = importlib.metadata.version("atmorad-py")
    except importlib.metadata.PackageNotFoundError:
        __version__ = "unknown"

    parser = argparse.ArgumentParser(prog="atmorad")
    parser.add_argument(
        "config", nargs="?", help="path to the simulation config TOML file", type=Path
    )
    parser.add_argument(
        "--init",
        action="store_true",
        help="generate a default configuration file in the current directory",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("-v", "--verbose", action="store_true", help="increase output verbosity")
    group.add_argument("-q", "--quiet", action="store_true", help="suppress all output")
    args = parser.parse_args()

    if args.init:
        init_config()
        sys.exit(0)

    if not args.config:
        parser.error("You must provide a configuration file, use --init to generate one.")

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

        for fig, relative_path in analyzer.generate_all_figures():
            data_io.save_figure(fig, relative_path)

        if not args.quiet:
            print("\n" + analyzer.summary())

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
