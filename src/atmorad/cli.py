import argparse
import importlib.metadata
import importlib.resources as pkg_resources
import logging
import sys
import traceback
from pathlib import Path


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
    parser.add_argument("--extract-config", help="path to a data.nc file", type=Path)

    group = parser.add_mutually_exclusive_group()
    group.add_argument("-v", "--verbose", action="store_true", help="increase output verbosity")
    group.add_argument(
        "-q", "--quiet", action="store_true", help="suppress all output except warnings and errors"
    )

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


def main():
    parser = setup_parser()
    args = parser.parse_args()

    try:
        if args.init:
            init_config()
            return 0

        if args.extract_config:
            from atmorad.output import DataIO

            DataIO.extract_config(data_path=args.extract_config)
            return 0

    except FileExistsError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if not args.config:
        parser.error("You must provide a configuration file, use --init to generate one.")

    configure_logging(args.verbose, args.quiet)

    try:
        import atmorad

        atmorad.run(args.config, args.quiet)
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
