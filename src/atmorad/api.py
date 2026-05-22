# src/atmorad/api.py
from pathlib import Path

from .builder import build_context
from .engine import MCRadiationRunner
from .output import DataIO, ResultAnalyzer


def save_all_figures(analyzer, data_io):
    for fig, relative_path in analyzer.generate_all_figures():
        data_io.save_figure(fig, relative_path)


def run(config_path: str | Path, quiet: bool = False) -> dict:
    """High-level API to run the simulation."""
    path = Path(config_path).resolve()
    context = build_context(path)

    data_io = DataIO(context.config)
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

    analyzer = ResultAnalyzer(results, context.config)
    save_all_figures(analyzer, data_io)

    if not quiet:
        print("\n".join((analyzer.experiment_summary(), data_io.output_summary())))

    return results
