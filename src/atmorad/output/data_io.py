import logging
import re
import shutil
from pathlib import Path

import xarray as xr
from matplotlib.figure import Figure

from atmorad.config import SimConfig
from atmorad.models.results import SimResults


class DataIO:
    """Handles all file system operations: saving results, config, and checkpoints."""

    NETCDF_ENGINE = "h5netcdf"

    def __init__(self, config: SimConfig) -> None:
        self.config = config
        meta = config.metadata

        self.exp_name = meta.experiment_name.replace(" ", "-")
        self.scen_name = meta.scenario_name
        self.results_filename = f"atmorad_{self.exp_name}_{self.scen_name}.nc"
        self.checkpoint_filename = f"{self.scen_name}-checkpoint.nc"

        output_dir = config.output.base_dir
        fig_dir = config.output.fig_dir
        timestamp = meta.run_timestamp
        resume = config.engine.resume_from_checkpoint
        overwrite = config.output.overwrite

        self.base_dir = output_dir / self.exp_name
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.results_filename = f"atmorad_{self.exp_name}_{self.scen_name}.nc"

        if resume:
            latest_checkpoint_dir = self._find_compatible_checkpoint_dir(output_dir)
            if latest_checkpoint_dir:
                self.base_dir = latest_checkpoint_dir
                self.fig_dir = fig_dir / latest_checkpoint_dir.name
                self.fig_dir.mkdir(parents=True, exist_ok=True)
                logging.info(f"Resuming from the most recent directory: {self.base_dir}")
                return

            logging.warning(
                f"Resume requested for '{self.exp_name}', but no checkpoint found. Starting fresh."
            )

        if overwrite:
            run_folder_name = self.exp_name
        else:
            run_folder_name = f"{self.exp_name}-{timestamp}"

        self.fig_dir = fig_dir / run_folder_name
        self.base_dir = output_dir / run_folder_name

        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.fig_dir.mkdir(parents=True, exist_ok=True)

    def output_summary(self) -> str:
        lines = [f"Outputs saved to: {self.base_dir}/"]
        files = [self.results_filename]
        for i, filename in enumerate(files):
            lines.append(f"  {'└─' if i == len(files) - 1 else '├─'} {filename}")
        return "\n".join(lines)

    def _generate_candidate_dirs(self, output_dir: Path):
        """Yields valid directories sorted from newest to oldest."""
        timestamp_pattern = re.compile(r"^\d{8}-\d{6}$")
        valid_dirs = []

        for candidate in output_dir.glob(f"{self.exp_name}-*"):
            if not candidate.is_dir():
                continue

            suffix = candidate.name.removeprefix(f"{self.exp_name}-")
            if not timestamp_pattern.fullmatch(suffix):
                continue

            if (candidate / self.checkpoint_filename).is_file() or (
                candidate / self.results_filename
            ).is_file():
                valid_dirs.append(candidate)

        base_dir = output_dir / self.exp_name
        if base_dir.is_dir():
            if (base_dir / self.checkpoint_filename).is_file() or (
                base_dir / self.results_filename
            ).is_file():
                valid_dirs.append(base_dir)

        valid_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        for d in valid_dirs:
            yield d

    def _find_compatible_checkpoint_dir(self, output_dir: Path) -> Path | None:
        """Iterates through candidate directories until a compatible one is found."""
        for candidate in self._generate_candidate_dirs(output_dir):
            file_to_check = candidate / self.results_filename
            if not file_to_check.exists():
                file_to_check = candidate / self.checkpoint_filename

            logging.debug(f"Checking compatibility for candidate: {candidate.name}")

            loaded_results = self._load_nc_file(file_to_check)

            if loaded_results is not None:
                if loaded_results.config is not None:
                    if self.config.is_compatible_for_resume(loaded_results.config):
                        logging.debug(f"Compatible checkpoint found in: {candidate.name}")
                        return candidate
                    else:
                        logging.debug(f"Skipped {candidate.name}: configuration mismatch.")

        return None

    def save_simulation_run(self, results: SimResults) -> None:
        results_path = self.base_dir / self.results_filename
        tmp_path = results_path.with_name(results_path.name + ".tmp")

        results.config = self.config
        ds = results.to_dataset(normalize=True)
        ds.to_netcdf(tmp_path, engine=self.NETCDF_ENGINE)

        shutil.move(tmp_path, results_path)

    def save_figure(self, fig: Figure, plot_name: str, dpi: int = 300) -> None:
        """Saves with prefix, e.g. plot_name="vertical_flux" -> "vertical_flux_demo001_baseline.png" """
        filename = f"{plot_name}_{self.exp_name}_{self.scen_name}.png"
        full_path = self.fig_dir / filename

        full_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(full_path, dpi=dpi, bbox_inches="tight")

    @property
    def checkpoint_path(self) -> Path:
        return self.base_dir / self.checkpoint_filename

    def save_checkpoint(self, results: SimResults) -> None:
        tmp_path = self.checkpoint_path.with_suffix(".nc.tmp")
        results.config = self.config
        ds = results.to_dataset(normalize=False)
        ds.to_netcdf(tmp_path, engine=self.NETCDF_ENGINE)
        shutil.move(tmp_path, self.checkpoint_path)

    def load_checkpoint(self) -> SimResults | None:
        """Loads simulation state from a completed results file or a checkpoint."""
        finished_path = self.base_dir / self.results_filename
        if finished_path.exists():
            return self._load_nc_file(finished_path)

        if self.checkpoint_path.exists():
            return self._load_nc_file(self.checkpoint_path)

        return None

    def _load_nc_file(self, path: Path) -> SimResults | None:
        if not path.exists():
            return None

        try:
            with xr.open_dataset(path, engine=self.NETCDF_ENGINE) as ds:
                ds.load()
                return SimResults.from_dataset(ds)
        except (OSError, ValueError):
            logging.exception("Failed to load checkpoint file.")
            return None

    def delete_checkpoint(self) -> None:
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()

    @classmethod
    def load_simulation_results(cls, directory: str | Path) -> SimResults:
        dir_path = Path(directory)

        nc_files = [f for f in dir_path.glob("*.nc") if "checkpoint" not in f.name]

        if not nc_files:
            raise FileNotFoundError(f"Could not find any result .nc files at {dir_path.resolve()}")
        elif len(nc_files) > 1:
            logging.warning(
                f"Multiple .nc files found in {dir_path}. Loading the most recently modified."
            )
            nc_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        results_path = nc_files[0]

        with xr.open_dataset(results_path, engine=cls.NETCDF_ENGINE) as ds:
            ds.load()
            return SimResults.from_dataset(ds)
