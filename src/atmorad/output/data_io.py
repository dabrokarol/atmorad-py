import datetime
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

    RESULTS_FILE = "data.nc"
    CONFIG_FILE = "runtime_config.toml"
    CHECKPOINT_FILE = "checkpoint.nc"
    NETCDF_ENGINE = "h5netcdf"
    FIG_DIR = "fig/"

    def __init__(self, config: SimConfig) -> None:
        self.config = config

        output_dir = Path(config.output.base_dir)
        exp_name = config.metadata.experiment_name.replace(" ", "-")
        resume = config.engine.resume_from_checkpoint
        overwrite = config.output.overwrite

        if resume:
            latest_checkpoint_dir = self._find_latest_checkpoint_dir(output_dir, exp_name)
            if latest_checkpoint_dir:
                self.base_dir = latest_checkpoint_dir
                logging.info(f"Resuming from the most recent directory: {self.base_dir}")
                return

            logging.warning(
                f"Resume requested for '{exp_name}', but no checkpoint found. Starting fresh."
            )

        if overwrite:
            self.base_dir = output_dir / f"{exp_name}"
            if self.base_dir.exists():
                shutil.rmtree(self.base_dir)
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.base_dir = output_dir / f"{exp_name}-{timestamp}"

        self.base_dir.mkdir(parents=True, exist_ok=True)

    def output_summary(self) -> str:
        """Generates a clean string summary of the saved file tree structure."""
        lines = [f"Outputs saved to: {self.base_dir}/"]
        files = [self.RESULTS_FILE, self.CONFIG_FILE]
        for i, filename in enumerate(files):
            if i == len(files) - 1:
                lines.append(f"  └─ {filename}")
            else:
                lines.append(f"  ├─ {filename}")

        return "\n".join(lines)

    def _find_latest_checkpoint_dir(self, output_dir: Path, exp_name: str) -> Path | None:
        """Return latest checkpoint dir for exp_name or exp_name-YYYYMMDD-HHMMSS."""
        timestamp_pattern = re.compile(r"^\d{8}-\d{6}$")
        valid_dirs = []
        for candidate in output_dir.glob(f"{exp_name}-*"):
            if not candidate.is_dir():
                continue
            suffix = candidate.name[len(exp_name) + 1 :]
            if not timestamp_pattern.fullmatch(suffix):
                continue
            if (candidate / self.CHECKPOINT_FILE).exists():
                valid_dirs.append(candidate)

        base_dir = output_dir / exp_name
        if base_dir.is_dir() and (base_dir / self.CHECKPOINT_FILE).exists():
            valid_dirs.append(base_dir)

        return (  # take the most recent from matching files
            max(valid_dirs, key=lambda p: (p / self.CHECKPOINT_FILE).stat().st_mtime)
            if valid_dirs
            else None
        )

    def save_config_file(self, config_file_path: Path) -> None:
        if not config_file_path.exists():
            logging.error(f"Cannot find original config at {config_file_path.resolve()}")
            return

        destination_path = self.base_dir / self.CONFIG_FILE
        shutil.copy2(config_file_path, destination_path)

    def save_simulation_run(self, results: SimResults) -> None:
        results_path = self.base_dir / self.RESULTS_FILE

        results.config = self.config
        ds = results.to_dataset(normalize=True)

        ds.to_netcdf(results_path, engine=self.NETCDF_ENGINE)

        if self.config.config_path:
            self.save_config_file(self.config.config_path)

    def save_figure(self, fig: Figure, relative_path: str, dpi: int = 300) -> None:
        full_path = self.base_dir / self.FIG_DIR / relative_path.lstrip("/")
        full_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(full_path, dpi=dpi, bbox_inches="tight")

    @property
    def checkpoint_path(self) -> Path:
        return self.base_dir / self.CHECKPOINT_FILE

    def save_checkpoint(self, results: SimResults) -> None:
        tmp_path = self.checkpoint_path.with_suffix(".nc.tmp")

        results.config = self.config
        ds = results.to_dataset(normalize=False)
        ds.to_netcdf(tmp_path, engine=self.NETCDF_ENGINE)
        shutil.move(tmp_path, self.checkpoint_path)

    def load_checkpoint(self) -> SimResults | None:
        """Loads checkpoint and returns a SimulationResults object, or None if not found."""
        if not self.checkpoint_path.exists():
            return None

        try:
            with xr.open_dataset(self.checkpoint_path, engine=self.NETCDF_ENGINE) as ds:
                ds.load()
                return SimResults.from_dataset(ds)
        except (OSError, ValueError):
            logging.exception("Failed to load checkpoint file.")
            return None

    def delete_checkpoint(self) -> None:
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()

    @classmethod
    def load_simulation_data(cls, directory: str | Path) -> SimResults:
        dir_path = Path(directory)
        results_path = dir_path / cls.RESULTS_FILE

        if not results_path.exists():
            raise FileNotFoundError(f"Could not find results at {results_path.resolve()}")

        with xr.open_dataset(results_path, engine=cls.NETCDF_ENGINE) as ds:
            ds.load()
            return SimResults.from_dataset(ds)
