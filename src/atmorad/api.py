import secrets
from pathlib import Path

from .builder import build_context_list
from .engine import MCRadiationRunner
from .models import SimResults
from .output import DataIO, ResultAnalyzer


def run(config_path: str | Path, quiet: bool = False) -> list[SimResults] | SimResults:
    path = Path(config_path).resolve()
    context_list = build_context_list(path)

    results_list = []
    random_seed = secrets.randbits(32)

    for context in context_list:
        data_io = DataIO(context.config)

        if context.config.engine.resume_from_checkpoint:
            if not data_io.checkpoint_config and not quiet:
                print(
                    f"[{context.config.metadata.scenario_name}] No compatible checkpoint found. Starting fresh."
                )

        if context.config.engine.random_seed == -1:
            if data_io.checkpoint_config:
                context.config.engine.random_seed = data_io.checkpoint_config.engine.random_seed
            else:
                context.config.engine.random_seed = random_seed

        runner = MCRadiationRunner(
            context,
            quiet=quiet,
            on_checkpoint=data_io.save_checkpoint,
            on_finish=data_io.save_simulation_run,
            load_checkpoint_fn=data_io.load_checkpoint,
            on_cleanup=data_io.delete_checkpoint,
        )

        runner.run()
        results = runner.get_results()
        results_list.append(results)

        analyzer = ResultAnalyzer(results.to_dataset(normalize=True))

        if context.config.output.save_plots:
            for fig, relative_path in analyzer.generate_all_figures():
                data_io.save_figure(fig, relative_path)

        if not quiet:
            print("\n".join((analyzer.experiment_summary(), data_io.output_summary())))

    return results_list[0] if len(results_list) == 1 else results_list


def load(directory: Path | str) -> SimResults:
    return DataIO.load_simulation_results(Path(directory).resolve())
