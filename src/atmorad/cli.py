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
        raise FileExistsError(
            f"Error: '{out_path}' already exists. Delete if you want to reinitialize.",
        )

    config_data = pkg_resources.files("atmorad.config").joinpath("simulation.toml").read_text()
    out_path.write_text(config_data)
    logging.info(f"Generated default configuration file at {out_path.resolve()}")


def setup_parser():
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

    return parser


def configure_logging(verbose, quiet):
    if quiet:
        level = logging.WARNING
        fmt = "%(levelname)s: %(message)s"
    elif verbose:
        level = logging.DEBUG
        fmt = "%(levelname)s: %(message)s"
    else:
        level = logging.INFO
        fmt = "%(message)s"

    logging.basicConfig(level=level, format=fmt)


def save_all_figures(analyzer, data_io):
    for fig, relative_path in analyzer.generate_all_figures():
        data_io.save_figure(fig, relative_path)


def run_simulation(config, quiet):
    config_path = config.resolve()

    logging.info(f"Loading configuration from: {config_path.name}")

    context = build_context(config_path)

    data_io = DataIO(context.config)

    logging.info(
        f"Starting {context.config.engine.cpu_cores}-core simulation ({context.config.engine.num_photons:_} photons)"
    )

    runner = MCRadiationRunner(
        context=context,
        quiet=quiet,
        on_checkpoint=data_io.save_checkpoint,
        on_finish=data_io.save_simulation_run,
        on_cleanup=data_io.delete_checkpoint,
        load_checkpoint_fn=data_io.load_checkpoint,
    )
    runner.run()

    results_dict = runner.get_results()
    analyzer = ResultAnalyzer(results_dict, context.config)

    save_all_figures(analyzer, data_io)

    if not quiet:
        print("\n".join((analyzer.experiment_summary(), data_io.output_summary())))


def main():
    parser = setup_parser()
    args = parser.parse_args()

    try:
        if args.init:
            init_config()
            return 0
    except FileExistsError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if not args.config:
        parser.error("You must provide a configuration file, use --init to generate one.")

    configure_logging(args.verbose, args.quiet)

    try:
        run_simulation(args.config, args.quiet)
        return 0

    except KeyboardInterrupt:
        print("\nSimulation aborted by user.", file=sys.stderr)
        return 130

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}", file=sys.stderr)

        if args.verbose:
            print("\n--- Detailed Stack Trace ---", file=sys.stderr)
            traceback.print_exc()
        else:
            print(
                "\n(Run with --verbose to see the full stack trace for debugging)", file=sys.stderr
            )

        return 1


if __name__ == "__main__":
    sys.exit(main())
