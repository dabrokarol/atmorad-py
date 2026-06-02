import secrets
from pathlib import Path

import xarray as xr

from atmorad.config.loader import load_scenarios
from atmorad.environment import Scene
from atmorad.output.io import DataIO, normalize_dataset
from atmorad.output.plotter import ResultAnalyzer
from atmorad.runner import execute_simulation


def run(config_path: str | Path, quiet: bool = False) -> xr.Dataset | list[xr.Dataset]:
    path = Path(config_path).resolve()
    config_list = load_scenarios(path)

    results_list = []
    random_seed = secrets.randbits(32)

    for config in config_list:
        scene = Scene.from_config(config)
        data_io = DataIO(config)

        initial_state = None
        if config.engine.resume_from_checkpoint:
            if not data_io.checkpoint_config and not quiet:
                print(
                    f"[{config.metadata.scenario_name}] No compatible checkpoint found. Starting fresh."
                )
            else:
                initial_state = data_io.load_checkpoint()

        if config.engine.random_seed == -1:
            if data_io.checkpoint_config:
                config.engine.random_seed = data_io.checkpoint_config.engine.random_seed
            else:
                config.engine.random_seed = random_seed

        results_ds = execute_simulation(
            config=config,
            scene=scene,
            initial_state=initial_state,
            quiet=quiet,
            on_checkpoint=data_io.save_checkpoint,
        )

        data_io.save_simulation_run(results_ds)
        data_io.delete_checkpoint()

        results_list.append(results_ds)
        norm_ds = normalize_dataset(results_ds)
        analyzer = ResultAnalyzer(norm_ds)

        if config.output.save_plots:
            for fig, relative_path in analyzer.generate_all_figures():
                data_io.save_figure(fig, relative_path)

        if not quiet:
            print("\n".join((analyzer.experiment_summary(), data_io.output_summary())))

    return results_list[0] if len(results_list) == 1 else results_list


def load(directory: Path | str) -> xr.Dataset:
    return DataIO.load_simulation_results(Path(directory).resolve())
